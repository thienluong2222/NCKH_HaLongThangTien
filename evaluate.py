#!/usr/bin/env python3
"""
Đánh giá mô hình phân loại lễ hội

Kịch bản đánh giá:
1. YOLO + Constraint: Chỉ dùng detection + Bayesian, không hỏi user
2. YOLO + Constraint + Perfect Oracle: Trả lời đúng theo JSON annotation
3. YOLO + Constraint + Favorable Oracle: Trả lời có lợi cho festival đúng

Metrics: Accuracy, Precision, Recall, F1 (overall + per festival)
"""

import os
import sys
import json
import glob
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# Thêm backend vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services import YOLOCSVPipeline, BayesianFestivalClassifier, ObjectDetection, sigmoid
from constraintsDB import CONSTRAINTS_DB, SUBCLASS_TO_FESTIVAL


class FestivalEvaluator:
    """Đánh giá mô hình phân loại lễ hội"""
    
    # Mapping tên thư mục → tên festival trong CONSTRAINTS_DB
    FOLDER_TO_FESTIVAL = {
        "Frame Chợ Nổi (dễ)": "Chợ nổi Cái Răng",
        "Frame Chợ Nổi (khó)": "Chợ nổi Cái Răng",
        "Frame Đờn Ca Tài Tử (dễ)": "Đờn ca tài tử",
        "Frame Đờn Ca Tài Tử (khó)": "Đờn ca tài tử",
        "Frame Dù Khê (dễ)": "Sân Khấu Dù Kê",
        "Frame Dù Khê (khó)": "Sân Khấu Dù Kê",
        "Frame Kỳ Yên (dễ)": "Lễ hội Kỳ Yên Đình Bình Thủy",
        "Frame Kỳ Yên (khó)": "Lễ hội Kỳ Yên Đình Bình Thủy",
        "Frame Nghinh Ông (dễ)": "Nghinh Ông",
        "Frame Nghinh Ông (khó)": "Nghinh Ông",
        "Frame Ngũ Âm (dễ)": "Nhạc Ngũ Âm người Khmer",
        "Frame Ngũ Âm (khó)": "Nhạc Ngũ Âm người Khmer",
        "Frame Ok Bom Boc (dễ)": "Ooc Bom Bóc",
        "Frame Ok Bom Boc (khó)": "Ooc Bom Bóc",
        "Frame Thác Côn (dễ)": "Lễ hội thác côn",
        "Frame Thác Côn (khó)": "Lễ hội thác côn",
        "Frame Chol Chnam Thmay (dễ)": "Tết Choi Chnam Thmay",
        "Frame Chol Chnam Thmay (khó)": "Tết Choi Chnam Thmay",
    }
    
    def __init__(self, model_path: str, csv_path: str, frame_dir: str, gemini_api_key: str = None):
        """
        Khởi tạo evaluator
        
        Args:
            model_path: Đường dẫn đến model YOLO
            csv_path: Đường dẫn đến CSV mapping
            frame_dir: Đường dẫn đến thư mục chứa Frame
            gemini_api_key: API key cho Gemini (optional, chỉ cần cho LLM)
        """
        print("=" * 60)
        print("🚀 KHỞI TẠO FESTIVAL EVALUATOR")
        print("=" * 60)
        
        self.frame_dir = frame_dir
        self.pipeline = YOLOCSVPipeline(model_path, csv_path)
        self.classifier = BayesianFestivalClassifier(gemini_api_key)
        
        # Danh sách festivals
        self.festivals = list(CONSTRAINTS_DB.keys())
        print(f"📋 Số lễ hội: {len(self.festivals)}")
        
        # Load test samples
        self.test_samples = self._load_test_samples()
        print(f"📊 Tổng mẫu test: {len(self.test_samples)}")
        
    def _load_test_samples(self):
        """Load tất cả test samples từ thư mục Frame"""
        samples = []
        
        for folder_name, festival_name in self.FOLDER_TO_FESTIVAL.items():
            folder_path = os.path.join(self.frame_dir, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"⚠️ Không tìm thấy: {folder_path}")
                continue
                
            # Tìm tất cả file JSON annotation
            json_files = glob.glob(os.path.join(folder_path, "*.json"))
            
            for json_path in json_files:
                # Tìm file ảnh tương ứng
                base_name = os.path.splitext(json_path)[0]
                image_path = None
                
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    candidate = base_name + ext
                    if os.path.exists(candidate):
                        image_path = candidate
                        break
                
                if image_path:
                    samples.append({
                        'image_path': image_path,
                        'json_path': json_path,
                        'ground_truth': festival_name,
                        'folder': folder_name
                    })
        
        # Thống kê per festival
        festival_counts = defaultdict(int)
        for s in samples:
            festival_counts[s['ground_truth']] += 1
        
        print("\n📈 Phân bố mẫu theo festival:")
        for fest, count in sorted(festival_counts.items()):
            print(f"   - {fest}: {count} mẫu")
            
        return samples
    
    def _extract_labels_from_json(self, json_path: str) -> set:
        """Trích xuất labels từ JSON annotation"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            labels = set()
            for shape in data.get('shapes', []):
                label = shape.get('label', '')
                if label:
                    labels.add(label)
            return labels
        except Exception as e:
            print(f"⚠️ Lỗi đọc JSON {json_path}: {e}")
            return set()
    
    def _get_festival_features(self, festival: str) -> set:
        """Lấy tất cả features thuộc về 1 festival"""
        features = set()
        
        # Từ CONSTRAINTS_DB
        if festival in CONSTRAINTS_DB:
            for rule in CONSTRAINTS_DB[festival]:
                features.update(rule[1])  # params
        
        # Từ SUBCLASS_TO_FESTIVAL (reverse lookup)
        for subclass, fests in SUBCLASS_TO_FESTIVAL.items():
            if festival in fests:
                features.add(subclass)
                
        return features
    
    def _convert_to_detections(self, raw_detections: list) -> list:
        """Chuyển raw detection dict thành ObjectDetection objects"""
        detections = []
        for d in raw_detections:
            if d.get('mapped'):
                obj = ObjectDetection(
                    subclass=d['label'],
                    confidence=d['confidence'],
                    frame_id=0,
                    time_stamp=0.0,
                    count=1,
                    bboxs=[d['box']]
                )
                detections.append(obj)
        return detections
    
    def _simulate_perfect_oracle(self, features: list, json_labels: set) -> dict:
        """
        Perfect Oracle: Trả lời đúng theo JSON annotation
        
        Args:
            features: List features cần hỏi
            json_labels: Set labels có trong JSON
            
        Returns:
            dict: {feature: {"status": bool, "confidence": float}}
        """
        result = {}
        for feature in features:
            # Normalize để so sánh
            feature_lower = feature.lower()
            has_feature = any(feature_lower in label.lower() or label.lower() in feature_lower 
                            for label in json_labels)
            
            result[feature] = {
                "status": has_feature,
                "confidence": 1.0 if has_feature else 0.0
            }
        return result
    
    def _simulate_favorable_oracle(self, features: list, json_labels: set, ground_truth: str) -> dict:
        """
        Favorable Oracle: Trả lời có lợi cho festival đúng
        
        Quy tắc:
        1. Feature CÓ trong ảnh → "có" (ground truth)
        2. Feature KHÔNG trong ảnh nhưng thuộc festival đúng → "có" (favorable)
        3. Feature KHÔNG trong ảnh và KHÔNG thuộc festival đúng → "không" (honest)
        
        Args:
            features: List features cần hỏi
            json_labels: Set labels có trong JSON
            ground_truth: Festival đúng
            
        Returns:
            dict: {feature: {"status": bool, "confidence": float}}
        """
        # Lấy features của festival đúng
        gt_features = self._get_festival_features(ground_truth)
        gt_features_lower = {f.lower() for f in gt_features}
        
        result = {}
        for feature in features:
            feature_lower = feature.lower()
            
            # Check 1: Feature có trong ảnh không?
            in_image = any(feature_lower in label.lower() or label.lower() in feature_lower 
                          for label in json_labels)
            
            # Check 2: Feature thuộc festival đúng không?
            belongs_to_gt = (feature_lower in gt_features_lower or 
                           any(feature_lower in f for f in gt_features_lower) or
                           any(f in feature_lower for f in gt_features_lower))
            
            if in_image:
                # Có trong ảnh → "có" (ground truth)
                result[feature] = {"status": True, "confidence": 1.0}
            elif belongs_to_gt:
                # Không trong ảnh nhưng thuộc GT → "có" (favorable)
                result[feature] = {"status": True, "confidence": 0.85}
            else:
                # Không trong ảnh và không thuộc GT → "không" (honest)
                result[feature] = {"status": False, "confidence": 0.0}
                
        return result
    
    def evaluate_single_sample(self, sample: dict, scenario: str = 'baseline') -> dict:
        """
        Đánh giá 1 sample
        
        Args:
            sample: Dict chứa image_path, json_path, ground_truth
            scenario: 'baseline' | 'perfect_oracle' | 'favorable_oracle'
            
        Returns:
            dict: Kết quả đánh giá
        """
        image_path = sample['image_path']
        json_path = sample['json_path']
        ground_truth = sample['ground_truth']
        
        # Step 1: YOLO Detection
        raw_dets = self.pipeline.predict_and_map_with_boxes(image_path, confidence_threshold=0.5)
        detections = self._convert_to_detections(raw_dets)
        
        # Step 2: Calculate initial logits
        logits, unsatisfied, satisfied = self.classifier.calculate_initial_logits(detections)
        
        # Step 3: Get prediction based on scenario
        if scenario == 'baseline':
            # Không hỏi thêm
            final_logits = logits
            num_turns = 0
        else:
            # Có hỏi thêm
            json_labels = self._extract_labels_from_json(json_path)
            
            # Select candidates
            candidates = self.classifier.select_candidates(logits)
            
            if not candidates:
                # Không có candidates → dùng baseline
                final_logits = logits
                num_turns = 0
            else:
                # Generate questions
                questions = self.classifier.generate_multi_turn_questions(candidates, unsatisfied)
                final_logits = logits.copy()
                num_turns = 0
                
                for q in questions:
                    features = q['target_features']
                    
                    # Simulate answer based on scenario
                    if scenario == 'perfect_oracle':
                        parsed_answer = self._simulate_perfect_oracle(features, json_labels)
                    else:  # favorable_oracle
                        parsed_answer = self._simulate_favorable_oracle(features, json_labels, ground_truth)
                    
                    # Update logits
                    final_logits = self.classifier.update_logits_from_consolidated_answer(
                        final_logits, candidates, unsatisfied, parsed_answer
                    )
                    num_turns += 1
                    
                    # Check if should continue
                    if not self.classifier.should_continue_asking(final_logits):
                        break
        
        # Step 4: Get prediction
        probs = {f: sigmoid(l) for f, l in final_logits.items()}
        predicted = max(probs, key=probs.get)
        confidence = probs[predicted]
        
        return {
            'ground_truth': ground_truth,
            'predicted': predicted,
            'correct': predicted == ground_truth,
            'confidence': confidence,
            'num_turns': num_turns,
            'all_probs': probs
        }
    
    def calculate_metrics(self, results: list) -> dict:
        """
        Tính toán metrics từ kết quả đánh giá
        
        Returns:
            dict: overall và per-festival metrics
        """
        # Overall
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        # Per-festival metrics
        festival_metrics = {}
        
        for festival in self.festivals:
            tp = sum(1 for r in results if r['ground_truth'] == festival and r['predicted'] == festival)
            fp = sum(1 for r in results if r['ground_truth'] != festival and r['predicted'] == festival)
            fn = sum(1 for r in results if r['ground_truth'] == festival and r['predicted'] != festival)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            support = sum(1 for r in results if r['ground_truth'] == festival)
            
            festival_metrics[festival] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        # Macro average
        macro_precision = np.mean([m['precision'] for m in festival_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in festival_metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in festival_metrics.values()])
        
        # Weighted average
        total_support = sum(m['support'] for m in festival_metrics.values())
        weighted_precision = sum(m['precision'] * m['support'] for m in festival_metrics.values()) / total_support if total_support > 0 else 0
        weighted_recall = sum(m['recall'] * m['support'] for m in festival_metrics.values()) / total_support if total_support > 0 else 0
        weighted_f1 = sum(m['f1'] * m['support'] for m in festival_metrics.values()) / total_support if total_support > 0 else 0
        
        # Average turns (for oracle scenarios)
        avg_turns = np.mean([r['num_turns'] for r in results])
        
        return {
            'overall': {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1,
                'avg_turns': avg_turns
            },
            'per_festival': festival_metrics
        }
    
    def run_evaluation(self, scenarios: list = None, verbose: bool = True) -> dict:
        """
        Chạy đánh giá với các kịch bản
        
        Args:
            scenarios: List kịch bản ['baseline', 'perfect_oracle', 'favorable_oracle']
            verbose: In chi tiết
            
        Returns:
            dict: Kết quả tất cả scenarios
        """
        if scenarios is None:
            scenarios = ['baseline', 'perfect_oracle', 'favorable_oracle']
        
        all_results = {}
        
        for scenario in scenarios:
            print(f"\n{'=' * 60}")
            print(f"🔄 ĐÁNH GIÁ KỊCH BẢN: {scenario.upper()}")
            print(f"{'=' * 60}")
            
            results = []
            for i, sample in enumerate(self.test_samples):
                if verbose and (i + 1) % 10 == 0:
                    print(f"   Đang xử lý: {i + 1}/{len(self.test_samples)}")
                
                result = self.evaluate_single_sample(sample, scenario)
                results.append(result)
            
            metrics = self.calculate_metrics(results)
            all_results[scenario] = {
                'results': results,
                'metrics': metrics
            }
            
            # Print summary
            self._print_metrics(scenario, metrics)
        
        # Print comparison table
        if len(scenarios) > 1:
            self._print_comparison_table(all_results)
        
        return all_results
    
    def _print_metrics(self, scenario: str, metrics: dict):
        """In metrics cho 1 scenario"""
        overall = metrics['overall']
        
        print(f"\n📊 KẾT QUẢ {scenario.upper()}:")
        print(f"   Accuracy:  {overall['accuracy']:.2%} ({overall['correct']}/{overall['total']})")
        print(f"   Macro F1:  {overall['macro_f1']:.4f}")
        print(f"   Weighted F1: {overall['weighted_f1']:.4f}")
        print(f"   Macro Precision: {overall['macro_precision']:.4f}")
        print(f"   Macro Recall: {overall['macro_recall']:.4f}")
        if overall['avg_turns'] > 0:
            print(f"   Avg Turns: {overall['avg_turns']:.2f}")
        
        print(f"\n   Per-Festival Metrics:")
        print(f"   {'Festival':<40} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
        print(f"   {'-' * 72}")
        
        for fest, m in sorted(metrics['per_festival'].items()):
            print(f"   {fest:<40} {m['precision']:>8.2%} {m['recall']:>8.2%} {m['f1']:>8.4f} {m['support']:>8}")
    
    def _print_comparison_table(self, all_results: dict):
        """In bảng so sánh giữa các scenarios"""
        print(f"\n{'=' * 80}")
        print("📋 BẢNG SO SÁNH CÁC KỊCH BẢN")
        print(f"{'=' * 80}")
        
        headers = ['Metric'] + list(all_results.keys())
        print(f"\n{'Metric':<25}", end='')
        for scenario in all_results.keys():
            print(f"{scenario:>20}", end='')
        print()
        print("-" * (25 + 20 * len(all_results)))
        
        metrics_to_compare = [
            ('Accuracy', lambda m: f"{m['overall']['accuracy']:.2%}"),
            ('Macro Precision', lambda m: f"{m['overall']['macro_precision']:.4f}"),
            ('Macro Recall', lambda m: f"{m['overall']['macro_recall']:.4f}"),
            ('Macro F1', lambda m: f"{m['overall']['macro_f1']:.4f}"),
            ('Weighted F1', lambda m: f"{m['overall']['weighted_f1']:.4f}"),
            ('Avg Turns', lambda m: f"{m['overall']['avg_turns']:.2f}"),
        ]
        
        for metric_name, get_value in metrics_to_compare:
            print(f"{metric_name:<25}", end='')
            for scenario, data in all_results.items():
                value = get_value(data['metrics'])
                print(f"{value:>20}", end='')
            print()
        
        # Improvement analysis
        if 'baseline' in all_results:
            baseline_acc = all_results['baseline']['metrics']['overall']['accuracy']
            
            print(f"\n📈 CẢI THIỆN SO VỚI BASELINE:")
            for scenario in all_results.keys():
                if scenario != 'baseline':
                    scenario_acc = all_results[scenario]['metrics']['overall']['accuracy']
                    abs_imp = scenario_acc - baseline_acc
                    rel_imp = (scenario_acc - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
                    print(f"   {scenario}: +{abs_imp:.2%} (tương đối: +{rel_imp:.1f}%)")
    
    def save_results(self, all_results: dict, output_dir: str = "evaluation_results"):
        """Lưu kết quả đánh giá"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = os.path.join(output_dir, f"evaluation_{timestamp}.json")
        
        # Convert results for JSON serialization
        json_data = {}
        for scenario, data in all_results.items():
            json_data[scenario] = {
                'metrics': data['metrics'],
                'sample_results': [
                    {
                        'ground_truth': r['ground_truth'],
                        'predicted': r['predicted'],
                        'correct': r['correct'],
                        'confidence': r['confidence'],
                        'num_turns': r['num_turns']
                    }
                    for r in data['results']
                ]
            }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Đã lưu kết quả: {json_path}")
        
        return json_path


def main():
    """Main function"""
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "backend/weight/best.pt")
    CSV_PATH = os.path.join(BASE_DIR, "backend/uploads/artifacts/merged_data.csv")
    FRAME_DIR = os.path.join(BASE_DIR, "assets/Frame")
    
    # Load API key from .env
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Check paths
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model: {MODEL_PATH}")
        return
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ Không tìm thấy CSV: {CSV_PATH}")
        return
    
    if not os.path.exists(FRAME_DIR):
        print(f"❌ Không tìm thấy Frame dir: {FRAME_DIR}")
        return
    
    # Create evaluator
    evaluator = FestivalEvaluator(
        model_path=MODEL_PATH,
        csv_path=CSV_PATH,
        frame_dir=FRAME_DIR,
        gemini_api_key=GEMINI_API_KEY
    )
    
    # Run evaluation
    all_results = evaluator.run_evaluation(
        scenarios=['baseline', 'perfect_oracle', 'favorable_oracle'],
        verbose=True
    )
    
    # Save results
    evaluator.save_results(all_results)
    
    print("\n✅ HOÀN THÀNH ĐÁNH GIÁ!")


if __name__ == "__main__":
    main()
