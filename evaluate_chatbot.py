"""
Đánh giá so sánh hiệu quả Chatbot.

So sánh 2 chế độ:
1. Không chatbot: YOLO → Logits → Kết luận ngay
2. Có chatbot: YOLO → Logits → Multi-turn Q&A → Kết luận

Sử dụng dữ liệu Frame với ground truth từ tên folder.
"""

import sys
import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse

# Thêm path để import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from constraintsDB import CONSTRAINTS_DB, SUBCLASS_TO_FESTIVAL


# ==========================================
# MAPPING VÀ CONFIG
# ==========================================

FOLDER_TO_FESTIVAL = {
    "Chợ Nổi": "Chợ nổi Cái Răng",
    "Chol Chnam Thmay": "Tết Choi Chnam Thmay",
    "Đờn Ca Tài Tử": "Đờn ca tài tử",
    "Dù Khê": "Sân Khấu Dù Kê",
    "Kỳ Yên": "Lễ hội Kỳ Yên Đình Bình Thủy",
    "Nghinh Ông": "Nghinh Ông",
    "Ngũ Âm": "Nhạc Ngũ Âm người Khmer",
    "Ok Bom Boc": "Ooc Bom Bóc",
    "Thác Côn": "Lễ hội thác côn",
}

# Normalize labels (JSON → YOLO format)
LABEL_NORMALIZE = {
    "Ao khoac vai": "Ao Khoac vai",
    "Moi doi": "Moi do",
    "hoa sen": "Hoa sen",
}


def get_festival_from_folder(folder_name):
    """Lấy ground truth festival từ tên folder."""
    for key, festival in FOLDER_TO_FESTIVAL.items():
        if key.lower() in folder_name.lower():
            return festival
    return None


def parse_labelme_json(json_path):
    """Parse LabelMe JSON để lấy các labels."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        labels = set()
        for shape in data.get('shapes', []):
            label = shape.get('label', '').strip()
            # Normalize
            label = LABEL_NORMALIZE.get(label, label)
            if label:
                labels.add(label)
        
        return labels
        
    except Exception as e:
        return set()


def simulate_answer(json_labels, target_features):
    """
    Simulate câu trả lời dựa trên JSON ground truth.
    
    Args:
        json_labels: Set các labels trong JSON
        target_features: List features cần hỏi
        
    Returns:
        dict: {feature: {"status": bool, "confidence": float}}
    """
    result = {}
    
    for feature in target_features:
        # Tìm kiếm case-insensitive và partial match
        found = any(
            feature.lower() == label.lower() or
            feature.lower() in label.lower() or
            label.lower() in feature.lower()
            for label in json_labels
        )
        
        result[feature] = {
            "status": found,
            "confidence": 1.0 if found else 0.9
        }
    
    return result


def convert_to_detections(raw_detections, frame_id=0):
    """
    Chuyển đổi raw detections từ YOLO pipeline sang ObjectDetection objects.
    
    Args:
        raw_detections: List[dict] từ predict_and_map
        frame_id: ID của frame
        
    Returns:
        List[ObjectDetection]
    """
    from services import ObjectDetection
    from collections import defaultdict
    
    # Group by subclass trong cùng frame
    grouped = defaultdict(list)
    for det in raw_detections:
        subclass = det.get('mapped_subclass') or det.get('detected_subclass')
        if subclass:
            grouped[subclass].append(det)
    
    detections = []
    for subclass, dets in grouped.items():
        # Tính average confidence
        avg_conf = sum(d.get('confidence', 0.5) for d in dets) / len(dets)
        
        # Lấy bboxes (nếu có)
        bboxs = []  # YOLO pipeline không trả về bbox trực tiếp trong matched results
        
        detection = ObjectDetection(
            subclass=subclass,
            confidence=avg_conf,
            frame_id=frame_id,
            time_stamp=frame_id / 30.0,  # Giả định 30fps
            count=len(dets),
            bboxs=bboxs
        )
        detections.append(detection)
    
    return detections


class ChatbotEvaluator:
    """Đánh giá so sánh accuracy có/không chatbot."""
    
    def __init__(self, model_path, csv_path, api_key=None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path tới YOLO weights
            csv_path: Path tới CSV mapping
            api_key: Gemini API key (optional for simulation)
        """
        from services import YOLOCSVPipeline, BayesianFestivalClassifier, sigmoid
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Load YOLO pipeline
        self.pipeline = YOLOCSVPipeline(model_path, csv_path)
        
        # Load Bayesian classifier (cần API key)
        if api_key is None:
            api_key = os.environ.get('GEMINI_API_KEY', '')
        
        if api_key:
            self.classifier = BayesianFestivalClassifier(api_key)
        else:
            print("⚠️ Không có GEMINI_API_KEY, một số chức năng sẽ bị giới hạn")
            self.classifier = None
        
        self.sigmoid = sigmoid
        
        # Statistics
        self.results_no_chat = []
        self.results_with_chat = []
        
    def evaluate_single_image(self, image_path, json_path, ground_truth):
        """
        Đánh giá một ảnh với cả 2 chế độ.
        
        Args:
            image_path: Path tới ảnh
            json_path: Path tới JSON annotation
            ground_truth: Festival ground truth
            
        Returns:
            dict: Kết quả so sánh
        """
        if not self.classifier:
            return {
                "image": os.path.basename(image_path),
                "ground_truth": ground_truth,
                "no_chat": {"predicted": None, "correct": False},
                "with_chat": {"predicted": None, "correct": False, "turns": 0},
                "error": "No classifier available"
            }
        
        # YOLO Detection
        raw_detections = self.pipeline.predict_and_map(
            image_path, show_image=False, confidence_threshold=0.5
        )
        
        if not raw_detections:
            return {
                "image": os.path.basename(image_path),
                "ground_truth": ground_truth,
                "no_chat": {"predicted": None, "correct": False},
                "with_chat": {"predicted": None, "correct": False, "turns": 0}
            }
        
        # Convert to ObjectDetection format
        detected_objects = convert_to_detections(raw_detections, frame_id=0)
        
        # Calculate initial logits
        logits, unsatisfied, satisfied = self.classifier.calculate_initial_logits(detected_objects)
        
        # ========== CHẾ ĐỘ 1: KHÔNG CHATBOT ==========
        probs_no_chat = {f: self.sigmoid(l) for f, l in logits.items()}
        predicted_no_chat = max(probs_no_chat.items(), key=lambda x: x[1])[0]
        correct_no_chat = predicted_no_chat == ground_truth
        
        # ========== CHẾ ĐỘ 2: CÓ CHATBOT ==========
        candidates = self.classifier.select_candidates(logits)
        
        current_logits = logits.copy()
        turns_used = 0
        
        if candidates:
            # Sinh câu hỏi
            questions = self.classifier.generate_multi_turn_questions(candidates, unsatisfied)
            
            # Lấy labels từ JSON để simulate
            json_labels = parse_labelme_json(json_path) if json_path else set()
            
            # Multi-turn Q&A
            for q in questions:
                turns_used += 1
                
                # Simulate câu trả lời
                parsed_answer = simulate_answer(json_labels, q['target_features'])
                
                # Cập nhật logits
                current_logits = self.classifier.update_logits_from_consolidated_answer(
                    current_logits, candidates, unsatisfied, parsed_answer
                )
                
                # Kiểm tra có cần hỏi tiếp không
                if not self.classifier.should_continue_asking(current_logits):
                    break
        
        # Kết luận với chatbot
        probs_with_chat = {f: self.sigmoid(l) for f, l in current_logits.items()}
        predicted_with_chat = max(probs_with_chat.items(), key=lambda x: x[1])[0]
        correct_with_chat = predicted_with_chat == ground_truth
        
        return {
            "image": os.path.basename(image_path),
            "ground_truth": ground_truth,
            "no_chat": {
                "predicted": predicted_no_chat,
                "confidence": probs_no_chat[predicted_no_chat],
                "correct": correct_no_chat
            },
            "with_chat": {
                "predicted": predicted_with_chat,
                "confidence": probs_with_chat[predicted_with_chat],
                "correct": correct_with_chat,
                "turns": turns_used
            }
        }
    
    def evaluate_folder(self, folder_path):
        """
        Đánh giá tất cả ảnh trong một folder.
        
        Args:
            folder_path: Path tới folder chứa frames
            
        Returns:
            list: Kết quả cho từng ảnh
        """
        folder = Path(folder_path)
        ground_truth = get_festival_from_folder(folder.name)
        
        if not ground_truth:
            print(f"⚠️ Không xác định được ground truth cho folder: {folder.name}")
            return []
        
        # Lấy các file ảnh và JSON
        image_files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        
        results = []
        
        for img_path in image_files:
            # Tìm file JSON tương ứng
            json_path = img_path.with_suffix('.json')
            if not json_path.exists():
                json_path = None
            
            result = self.evaluate_single_image(
                str(img_path),
                str(json_path) if json_path else None,
                ground_truth
            )
            results.append(result)
        
        return results
    
    def evaluate_all_folders(self, base_path, max_images_per_folder=10):
        """
        Đánh giá tất cả folders.
        
        Args:
            base_path: Path tới thư mục chứa các folder Frame
            max_images_per_folder: Số ảnh tối đa để test mỗi folder
        """
        base = Path(base_path)
        all_folders = [f for f in base.iterdir() if f.is_dir() and "Frame " in f.name]
        
        print(f"\n📁 Tìm thấy {len(all_folders)} folders")
        print("=" * 70)
        
        all_results = []
        
        from tqdm import tqdm
        
        for folder in tqdm(all_folders, desc="Đánh giá folders"):
            ground_truth = get_festival_from_folder(folder.name)
            if not ground_truth:
                continue
            
            # Lấy ảnh
            image_files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
            image_files = image_files[:max_images_per_folder]  # Giới hạn
            
            for img_path in image_files:
                json_path = img_path.with_suffix('.json')
                
                try:
                    result = self.evaluate_single_image(
                        str(img_path),
                        str(json_path) if json_path.exists() else None,
                        ground_truth
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"⚠️ Lỗi với {img_path.name}: {e}")
        
        return all_results
    
    def calculate_metrics(self, results):
        """
        Tính toán các metrics so sánh.
        
        Args:
            results: List kết quả từ evaluate
            
        Returns:
            dict: Metrics tổng hợp
        """
        if not results:
            return {}
        
        # Overall metrics
        correct_no_chat = sum(1 for r in results if r['no_chat']['correct'])
        correct_with_chat = sum(1 for r in results if r['with_chat']['correct'])
        total = len(results)
        
        accuracy_no_chat = correct_no_chat / total
        accuracy_with_chat = correct_with_chat / total
        improvement = accuracy_with_chat - accuracy_no_chat
        
        # Per-festival metrics
        per_festival = defaultdict(lambda: {
            "total": 0,
            "correct_no_chat": 0,
            "correct_with_chat": 0,
            "improved": 0,  # Số case được sửa đúng nhờ chatbot
            "worsened": 0,  # Số case bị sai thêm do chatbot
        })
        
        for r in results:
            gt = r['ground_truth']
            per_festival[gt]["total"] += 1
            
            if r['no_chat']['correct']:
                per_festival[gt]["correct_no_chat"] += 1
            
            if r['with_chat']['correct']:
                per_festival[gt]["correct_with_chat"] += 1
            
            # Track changes
            if not r['no_chat']['correct'] and r['with_chat']['correct']:
                per_festival[gt]["improved"] += 1
            elif r['no_chat']['correct'] and not r['with_chat']['correct']:
                per_festival[gt]["worsened"] += 1
        
        # Convert to regular dict with accuracy
        per_festival_metrics = {}
        for fest, data in per_festival.items():
            per_festival_metrics[fest] = {
                "total": data["total"],
                "accuracy_no_chat": data["correct_no_chat"] / data["total"] if data["total"] > 0 else 0,
                "accuracy_with_chat": data["correct_with_chat"] / data["total"] if data["total"] > 0 else 0,
                "improved": data["improved"],
                "worsened": data["worsened"],
            }
        
        # Average turns
        avg_turns = sum(r['with_chat']['turns'] for r in results) / total
        
        return {
            "total_samples": total,
            "accuracy_no_chat": accuracy_no_chat,
            "accuracy_with_chat": accuracy_with_chat,
            "improvement": improvement,
            "improvement_percent": (improvement / accuracy_no_chat * 100) if accuracy_no_chat > 0 else 0,
            "correct_no_chat": correct_no_chat,
            "correct_with_chat": correct_with_chat,
            "avg_turns": avg_turns,
            "per_festival": per_festival_metrics
        }
    
    def print_report(self, metrics):
        """In báo cáo so sánh."""
        print("\n" + "=" * 70)
        print("📊 BÁO CÁO SO SÁNH HIỆU QUẢ CHATBOT")
        print("=" * 70)
        
        print(f"\n📈 TỔNG QUAN:")
        print(f"   Tổng số samples: {metrics['total_samples']}")
        print(f"   Average turns:   {metrics['avg_turns']:.2f}")
        
        print(f"\n┌{'─' * 50}┐")
        print(f"│{'KHÔNG CHATBOT':^50}│")
        print(f"├{'─' * 50}┤")
        print(f"│   Accuracy: {metrics['accuracy_no_chat']:.2%} ({metrics['correct_no_chat']}/{metrics['total_samples']}){' ' * 15}│")
        print(f"└{'─' * 50}┘")
        
        print(f"\n┌{'─' * 50}┐")
        print(f"│{'CÓ CHATBOT':^50}│")
        print(f"├{'─' * 50}┤")
        print(f"│   Accuracy: {metrics['accuracy_with_chat']:.2%} ({metrics['correct_with_chat']}/{metrics['total_samples']}){' ' * 15}│")
        print(f"└{'─' * 50}┘")
        
        print(f"\n🎯 CẢI THIỆN:")
        improvement_sign = "+" if metrics['improvement'] >= 0 else ""
        print(f"   Absolute: {improvement_sign}{metrics['improvement']:.2%}")
        print(f"   Relative: {improvement_sign}{metrics['improvement_percent']:.1f}%")
        
        print(f"\n📊 CHI TIẾT THEO LỄ HỘI:")
        print("-" * 70)
        print(f"{'Lễ hội':<35} {'No Chat':>10} {'With Chat':>10} {'Δ':>8}")
        print("-" * 70)
        
        for fest, data in sorted(metrics['per_festival'].items()):
            acc_no = f"{data['accuracy_no_chat']:.1%}"
            acc_with = f"{data['accuracy_with_chat']:.1%}"
            delta = data['accuracy_with_chat'] - data['accuracy_no_chat']
            delta_str = f"{'+' if delta >= 0 else ''}{delta:.1%}"
            
            print(f"{fest:<35} {acc_no:>10} {acc_with:>10} {delta_str:>8}")
            
            # Chi tiết improved/worsened
            if data['improved'] > 0 or data['worsened'] > 0:
                print(f"   {'':35} ↑{data['improved']} improved, ↓{data['worsened']} worsened")
        
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chatbot Effectiveness")
    parser.add_argument("--frame-path", "-f", 
                       default="assets/Frame",
                       help="Path to Frame folders")
    parser.add_argument("--max-images", "-m", 
                       type=int, default=5,
                       help="Max images per folder (default: 5)")
    parser.add_argument("--output", "-o",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🚀 CHATBOT EFFECTIVENESS EVALUATION")
    print("=" * 70)
    
    # Paths
    model_path = "weight/best.pt"
    csv_path = "artifacts/merged_data.csv"
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model: {model_path}")
        return
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy CSV: {csv_path}")
        return
    
    # Initialize evaluator
    print(f"\n📦 Loading model...")
    evaluator = ChatbotEvaluator(model_path, csv_path)
    
    # Run evaluation
    print(f"\n🔍 Đánh giá folders trong: {args.frame_path}")
    print(f"   Max images per folder: {args.max_images}")
    
    results = evaluator.evaluate_all_folders(args.frame_path, args.max_images)
    
    if not results:
        print("❌ Không có kết quả nào!")
        return
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(results)
    
    # Print report
    evaluator.print_report(metrics)
    
    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "frame_path": args.frame_path,
                "max_images": args.max_images
            },
            "metrics": metrics,
            "detailed_results": results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Đã lưu kết quả vào: {args.output}")


if __name__ == "__main__":
    main()
