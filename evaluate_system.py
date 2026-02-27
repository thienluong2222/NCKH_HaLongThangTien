"""
Đánh giá hệ thống nhận dạng lễ hội
=====================================
Script đánh giá Precision, Recall, F1-score, Accuracy
cho hệ thống YOLO + Bayesian Constraints

Flow:
1. YOLO detect từ ảnh JPG
2. Bayesian constraints tính logits → predict festival
3. So sánh với ground truth (tên folder) → tính metrics

Dữ liệu: assets/Frame/
Output: evaluation_results/evaluation_YYYYMMDD_HHMMSS.txt
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# Thêm backend vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services import YOLOCSVPipeline, ObjectDetection
from constraintsDB import CONSTRAINTS_DB


def sigmoid(x):
    """Hàm sigmoid để chuyển logit thành probability"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# ==========================================
# CONSTRAINT CHECKER (standalone, không cần LLM)
# ==========================================
class SimpleConstraintChecker:
    """
    Simplified Bayesian constraint checker for evaluation.
    Không sử dụng LLM - chỉ tính logits dựa trên constraints.
    """
    
    def _index_detections(self, detections):
        by_subclass = defaultdict(list)
        by_frame = defaultdict(list)
        for d in detections:
            by_subclass[d.subclass].append(d)
            by_frame[d.frame_id].append(d)
        return by_subclass, by_frame

    def _check_is_on(self, top_sub, bot_sub, by_subclass, by_frame):
        relevant_frames = set(d.frame_id for d in by_subclass[top_sub]) & set(d.frame_id for d in by_subclass[bot_sub])
        for fid in relevant_frames:
            tops = [d for d in by_frame[fid] if d.subclass == top_sub]
            bots = [d for d in by_frame[fid] if d.subclass == bot_sub]
            for t in tops:
                for b in bots:
                    for box_t in t.bboxs:
                        for box_b in b.bboxs:
                            x_overlap = max(0, min(box_t[2], box_b[2]) - max(box_t[0], box_b[0]))
                            width_t = box_t[2] - box_t[0]
                            vertical_gap = box_b[1] - box_t[3]
                            if width_t > 0 and (x_overlap/width_t) > 0.3 and -50 <= vertical_gap <= 50:
                                return True
        return False

    def check_constraints(self, rule, by_subclass, by_frame):
        ctype, params, is_hard, weight, threshold = rule
        satisfied = False
        if ctype == "is_presence":
            satisfied = len([p for p in params if p not in by_subclass]) == 0
        elif ctype == "is_presence_in_frame":
            for fid, dets in by_frame.items():
                subs = {d.subclass for d in dets}
                if all(p in subs for p in params):
                    satisfied = True; break
        elif ctype == "at_least":
            total = sum(sum(d.count for d in by_subclass[p]) for p in params if p in by_subclass)
            satisfied = total >= (threshold or 1)
        elif ctype == "at_least_in_frame":
            for fid, dets in by_frame.items():
                cnt = sum(d.count for d in dets if d.subclass in params)
                if cnt >= (threshold or 1):
                    satisfied = True; break
        elif ctype == "confidence_min":
            target = list(by_subclass.keys()) if "all" in params else [p for p in params if p in by_subclass]
            total_weighted_conf = 0
            total_count = 0
            for s in target:
                for d in by_subclass[s]:
                    total_weighted_conf += d.confidence * d.count
                    total_count += d.count
            if total_count > 0:
                avg = total_weighted_conf / total_count
                satisfied = avg >= (threshold or 0)
            else:
                satisfied = False
        elif ctype == "is_on" and len(params) == 2:
            satisfied = self._check_is_on(params[0], params[1], by_subclass, by_frame)
        return satisfied

    def calculate_logits(self, detections):
        """
        Tính logits cho mỗi lễ hội dựa trên detections
        """
        by_subclass, by_frame = self._index_detections(detections)
        
        festival_logits = {}
        festival_satisfied = defaultdict(list)
        festival_unsatisfied = defaultdict(list)

        for festival, rules in CONSTRAINTS_DB.items():
            current_logit = 0.0
            for rule in rules:
                is_satisfied = self.check_constraints(rule, by_subclass, by_frame)
                weight = rule[3]
                if is_satisfied:
                    current_logit += weight
                    festival_satisfied[festival].append(rule)
                else:
                    festival_unsatisfied[festival].append(rule)
            festival_logits[festival] = current_logit
        
        return festival_logits, festival_unsatisfied, festival_satisfied


# ==========================================
# MAPPING TÊN THƯ MỤC -> TÊN LỄ HỘI
# ==========================================
FOLDER_TO_FESTIVAL = {
    "Frame Chợ Nổi": "Chợ nổi Cái Răng",
    "Frame Ngũ Âm": "Nhạc Ngũ Âm người Khmer",
    "Frame Dù Khê": "Sân Khấu Dù Kê",
    "Frame Nghinh Ông": "Nghinh Ông",
    "Frame Kỳ Yên": "Lễ hội Kỳ Yên Đình Bình Thủy",
    "Frame Thác Côn": "Lễ hội thác côn",
    "Frame Ok Bom Boc": "Ooc Bom Bóc",
    "Frame Chol Chnam Thmay": "Tết Choi Chnam Thmay",
    "Frame Đờn Ca Tài Tử": "Đờn ca tài tử",
}


class SystemEvaluator:
    def __init__(self, frame_dir, model_path, csv_path):
        """
        Khởi tạo evaluator
        
        Args:
            frame_dir: Đường dẫn đến thư mục chứa frames (assets/Frame)
            model_path: Đường dẫn đến YOLO model
            csv_path: Đường dẫn đến CSV mapping
        """
        self.frame_dir = frame_dir
        
        print("=" * 70)
        print("🚀 KHỞI TẠO HỆ THỐNG ĐÁNH GIÁ")
        print("=" * 70)
        
        # Khởi tạo YOLO pipeline
        print("\n📦 Đang load YOLO model...")
        self.yolo_pipe = YOLOCSVPipeline(model_path, csv_path)
        
        # Khởi tạo Constraint Checker (không cần LLM)
        print("📦 Đang khởi tạo Constraint Checker...")
        self.checker = SimpleConstraintChecker()
        
        # Kết quả
        self.results = []
        self.results_by_difficulty = {'easy': [], 'hard': []}
        self.results_by_festival = defaultdict(list)
        
        # Object detection metrics
        self.all_detection_metrics = []
        
        print("\n✅ Khởi tạo hoàn tất!")
        print("=" * 70)
    
    def parse_folder_name(self, folder_name):
        """
        Parse tên thư mục để lấy tên lễ hội và độ khó
        """
        difficulty = None
        base_name = folder_name
        
        if "(dễ)" in folder_name:
            difficulty = "easy"
            base_name = folder_name.replace("(dễ)", "").strip()
        elif "(khó)" in folder_name:
            difficulty = "hard"
            base_name = folder_name.replace("(khó)", "").strip()
        
        festival_name = FOLDER_TO_FESTIVAL.get(base_name)
        return festival_name, difficulty
    
    def load_ground_truth_from_json(self, json_path):
        """
        Load ground truth labels từ file JSON
        
        Returns:
            set: Tập các labels trong ground truth
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        labels = set()
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            if label:
                labels.add(label)
        
        return labels
    
    def detect_from_image(self, image_path, frame_id=0):
        """
        Chạy YOLO detection trên ảnh
        
        Args:
            image_path: Đường dẫn ảnh
            frame_id: ID của frame
            
        Returns:
            List[ObjectDetection]: Danh sách detections
        """
        # Chạy YOLO prediction
        results = self.yolo_pipe.model.predict(image_path, verbose=False)
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return []
        
        # Group detections by subclass
        detection_groups = defaultdict(lambda: {'confidences': [], 'bboxs': []})
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            # Lấy tên class từ model (đây chính là subclass)
            subclass = self.yolo_pipe.model.names[cls_id]
            
            detection_groups[subclass]['confidences'].append(conf)
            detection_groups[subclass]['bboxs'].append(bbox)
        
        # Tạo ObjectDetection cho mỗi subclass
        detections = []
        for subclass, data in detection_groups.items():
            obj = ObjectDetection(
                subclass=subclass,
                confidence=np.mean(data['confidences']),
                frame_id=frame_id,
                time_stamp=0.0,
                count=len(data['confidences']),
                bboxs=data['bboxs']
            )
            detections.append(obj)
        
        return detections
    
    def calculate_detection_metrics(self, yolo_labels, gt_labels):
        """
        Tính Precision/Recall/F1 cho object detection (level subclass)
        
        Args:
            yolo_labels: set - Labels từ YOLO detection
            gt_labels: set - Labels từ ground truth (JSON)
            
        Returns:
            dict: Metrics cho detection
        """
        tp = len(yolo_labels & gt_labels)  # True Positives
        fp = len(yolo_labels - gt_labels)  # False Positives  
        fn = len(gt_labels - yolo_labels)  # False Negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'yolo_labels': yolo_labels,
            'gt_labels': gt_labels
        }
    
    def evaluate_single_folder(self, folder_path, ground_truth_festival):
        """
        Đánh giá một folder (nhiều ảnh của cùng 1 lễ hội)
        
        Args:
            folder_path: Đường dẫn folder
            ground_truth_festival: Tên lễ hội thực tế (ground truth)
            
        Returns:
            dict: Kết quả đánh giá
        """
        # Lấy tất cả file jpg và json
        all_files = os.listdir(folder_path)
        jpg_files = sorted([f for f in all_files if f.endswith('.jpg')])
        
        all_detections = []
        all_yolo_labels = set()
        all_gt_labels = set()
        per_image_metrics = []
        
        for i, jpg_file in enumerate(jpg_files):
            jpg_path = os.path.join(folder_path, jpg_file)
            json_path = jpg_path.replace('.jpg', '.json')
            
            # YOLO detection
            detections = self.detect_from_image(jpg_path, frame_id=i)
            for det in detections:
                all_detections.append(det)
                all_yolo_labels.add(det.subclass)
            
            # Load ground truth nếu có JSON
            if os.path.exists(json_path):
                gt_labels = self.load_ground_truth_from_json(json_path)
                all_gt_labels.update(gt_labels)
                
                # Tính metrics cho từng ảnh
                yolo_labels_this_img = set(det.subclass for det in detections)
                img_metrics = self.calculate_detection_metrics(yolo_labels_this_img, gt_labels)
                per_image_metrics.append(img_metrics)
        
        # Tính detection metrics tổng hợp cho folder
        folder_detection_metrics = self.calculate_detection_metrics(all_yolo_labels, all_gt_labels)
        
        # Tính Bayesian logits từ YOLO detections
        if all_detections:
            logits, unsatisfied, satisfied = self.checker.calculate_logits(all_detections)
            probs = {f: sigmoid(l) for f, l in logits.items()}
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            prediction = sorted_probs[0][0] if sorted_probs else None
            confidence = sorted_probs[0][1] if sorted_probs else 0.0
            top_3 = [(f, round(float(p), 4)) for f, p in sorted_probs[:3]]
        else:
            prediction = None
            confidence = 0.0
            top_3 = []
        
        # Kiểm tra festival prediction đúng/sai
        correct = (prediction == ground_truth_festival)
        
        return {
            'ground_truth': ground_truth_festival,
            'prediction': prediction,
            'top_3': top_3,
            'confidence': confidence,
            'correct': correct,
            'num_images': len(jpg_files),
            'yolo_labels': list(all_yolo_labels),
            'gt_labels': list(all_gt_labels),
            'detection_metrics': folder_detection_metrics,
            'per_image_metrics': per_image_metrics
        }
    
    def run_evaluation(self):
        """
        Chạy đánh giá trên toàn bộ dataset
        """
        print("\n" + "=" * 70)
        print("🔍 BẮT ĐẦU ĐÁNH GIÁ")
        print("=" * 70)
        
        folders = [f for f in os.listdir(self.frame_dir) 
                   if os.path.isdir(os.path.join(self.frame_dir, f)) and not f.startswith('_')]
        
        print(f"\n📁 Tìm thấy {len(folders)} thư mục")
        
        for folder_name in sorted(folders):
            folder_path = os.path.join(self.frame_dir, folder_name)
            
            # Parse tên thư mục
            festival_name, difficulty = self.parse_folder_name(folder_name)
            
            if not festival_name:
                print(f"⚠️  Bỏ qua thư mục không nhận diện được: {folder_name}")
                continue
            
            print(f"\n📂 Đang đánh giá: {folder_name}")
            print(f"   └─ Lễ hội: {festival_name}, Độ khó: {difficulty}")
            
            # Đánh giá
            result = self.evaluate_single_folder(folder_path, festival_name)
            result['folder'] = folder_name
            result['difficulty'] = difficulty
            
            self.results.append(result)
            self.all_detection_metrics.append(result['detection_metrics'])
            
            if difficulty:
                self.results_by_difficulty[difficulty].append(result)
            self.results_by_festival[festival_name].append(result)
            
            # In kết quả ngắn gọn
            status = "✅" if result['correct'] else "❌"
            det_metrics = result['detection_metrics']
            print(f"   └─ {status} Prediction: {result['prediction']} ({result['confidence']:.2%})")
            print(f"   └─ Detection: P={det_metrics['precision']:.2f}, R={det_metrics['recall']:.2f}, F1={det_metrics['f1']:.2f}")
            print(f"   └─ YOLO detected: {len(result['yolo_labels'])} classes, GT: {len(result['gt_labels'])} classes")
        
        return self.results
    
    def calculate_festival_metrics(self, results):
        """
        Tính metrics cho Festival Classification (Accuracy, Precision, Recall, F1)
        """
        if not results:
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'total': 0, 'correct': 0, 'per_class': {}
            }
        
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / total if total > 0 else 0.0
        
        # Macro averaging cho multi-class
        festivals = list(CONSTRAINTS_DB.keys())
        precision_sum = 0.0
        recall_sum = 0.0
        valid_classes = 0
        per_class_metrics = {}
        
        for festival in festivals:
            tp = sum(1 for r in results if r['prediction'] == festival and r['ground_truth'] == festival)
            fp = sum(1 for r in results if r['prediction'] == festival and r['ground_truth'] != festival)
            fn = sum(1 for r in results if r['prediction'] != festival and r['ground_truth'] == festival)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if tp + fp + fn > 0:
                precision_sum += precision
                recall_sum += recall
                valid_classes += 1
                per_class_metrics[festival] = {
                    'precision': precision, 'recall': recall, 'f1_score': f1,
                    'tp': tp, 'fp': fp, 'fn': fn
                }
        
        macro_precision = precision_sum / valid_classes if valid_classes > 0 else 0.0
        macro_recall = recall_sum / valid_classes if valid_classes > 0 else 0.0
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy, 'precision': macro_precision, 'recall': macro_recall, 'f1_score': macro_f1,
            'total': total, 'correct': correct, 'per_class': per_class_metrics
        }
    
    def calculate_overall_detection_metrics(self, metrics_list):
        """
        Tính detection metrics tổng hợp (micro-average)
        """
        total_tp = sum(m['tp'] for m in metrics_list)
        total_fp = sum(m['fp'] for m in metrics_list)
        total_fn = sum(m['fn'] for m in metrics_list)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision, 'recall': recall, 'f1': f1,
            'total_tp': total_tp, 'total_fp': total_fp, 'total_fn': total_fn
        }
    
    def generate_report(self):
        """
        Tạo báo cáo đánh giá chi tiết
        """
        report_lines = []
        
        def add_line(text=""):
            report_lines.append(text)
        
        def add_separator(char="=", length=70):
            add_line(char * length)
        
        # Header
        add_separator()
        add_line("BÁO CÁO ĐÁNH GIÁ HỆ THỐNG NHẬN DẠNG LỄ HỘI")
        add_line("(YOLO Detection + Bayesian Constraints)")
        add_separator()
        add_line(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        add_line(f"Thư mục dữ liệu: {self.frame_dir}")
        add_line()
        
        # =============================================
        # PHẦN 1: OBJECT DETECTION METRICS (YOLO)
        # =============================================
        add_separator("-")
        add_line("📊 PHẦN 1: ĐÁNH GIÁ YOLO OBJECT DETECTION")
        add_line("(So sánh labels YOLO detect với ground truth JSON)")
        add_separator("-")
        
        det_overall = self.calculate_overall_detection_metrics(self.all_detection_metrics)
        add_line(f"\n🎯 METRICS TỔNG HỢP (Micro-Average):")
        add_line(f"   • Precision: {det_overall['precision']:.4f} ({det_overall['precision']*100:.2f}%)")
        add_line(f"   • Recall:    {det_overall['recall']:.4f} ({det_overall['recall']*100:.2f}%)")
        add_line(f"   • F1-Score:  {det_overall['f1']:.4f} ({det_overall['f1']*100:.2f}%)")
        add_line(f"   • TP={det_overall['total_tp']}, FP={det_overall['total_fp']}, FN={det_overall['total_fn']}")
        add_line()
        
        # Detection theo độ khó
        add_line("📊 DETECTION THEO ĐỘ KHÓ:")
        for difficulty, label in [('easy', 'DỄ'), ('hard', 'KHÓ')]:
            results = self.results_by_difficulty.get(difficulty, [])
            if results:
                det_metrics = [r['detection_metrics'] for r in results]
                det_agg = self.calculate_overall_detection_metrics(det_metrics)
                add_line(f"\n   🔹 {label} ({len(results)} folders):")
                add_line(f"      • Precision: {det_agg['precision']:.4f}")
                add_line(f"      • Recall:    {det_agg['recall']:.4f}")
                add_line(f"      • F1-Score:  {det_agg['f1']:.4f}")
        add_line()
        
        # =============================================
        # PHẦN 2: FESTIVAL CLASSIFICATION METRICS
        # =============================================
        add_separator("-")
        add_line("📊 PHẦN 2: ĐÁNH GIÁ FESTIVAL CLASSIFICATION")
        add_line("(Bayesian Constraints từ YOLO detections)")
        add_separator("-")
        
        overall_metrics = self.calculate_festival_metrics(self.results)
        
        add_line(f"\nTổng số mẫu: {overall_metrics['total']}")
        add_line(f"Số mẫu đúng: {overall_metrics['correct']}")
        add_line()
        add_line(f"📈 METRICS TỔNG THỂ (Macro Average):")
        add_line(f"   • Accuracy:  {overall_metrics['accuracy']:.4f} ({overall_metrics['accuracy']*100:.2f}%)")
        add_line(f"   • Precision: {overall_metrics['precision']:.4f} ({overall_metrics['precision']*100:.2f}%)")
        add_line(f"   • Recall:    {overall_metrics['recall']:.4f} ({overall_metrics['recall']*100:.2f}%)")
        add_line(f"   • F1-Score:  {overall_metrics['f1_score']:.4f} ({overall_metrics['f1_score']*100:.2f}%)")
        add_line()
        
        # Kết quả theo độ khó
        add_line("📊 CLASSIFICATION THEO ĐỘ KHÓ:")
        for difficulty, label in [('easy', 'DỄ'), ('hard', 'KHÓ')]:
            results = self.results_by_difficulty.get(difficulty, [])
            if results:
                metrics = self.calculate_festival_metrics(results)
                add_line(f"\n   🔹 {label} ({len(results)} mẫu):")
                add_line(f"      • Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
                add_line(f"      • Precision: {metrics['precision']:.4f}")
                add_line(f"      • Recall:    {metrics['recall']:.4f}")
                add_line(f"      • F1-Score:  {metrics['f1_score']:.4f}")
        add_line()
        
        # Kết quả theo từng lễ hội
        add_separator("-")
        add_line("📊 PHẦN 3: KẾT QUẢ THEO TỪNG LỄ HỘI")
        add_separator("-")
        
        for festival, results in sorted(self.results_by_festival.items()):
            metrics = self.calculate_festival_metrics(results)
            det_metrics = [r['detection_metrics'] for r in results]
            det_agg = self.calculate_overall_detection_metrics(det_metrics)
            
            add_line(f"\n🎭 {festival}")
            add_line(f"   Số mẫu: {len(results)}, Classification đúng: {metrics['correct']}")
            add_line(f"   [Detection]  P={det_agg['precision']:.2f}, R={det_agg['recall']:.2f}, F1={det_agg['f1']:.2f}")
            
            if festival in overall_metrics.get('per_class', {}):
                pc = overall_metrics['per_class'][festival]
                add_line(f"   [Classification] P={pc['precision']:.2f}, R={pc['recall']:.2f}, F1={pc['f1_score']:.2f}")
                add_line(f"   TP={pc['tp']}, FP={pc['fp']}, FN={pc['fn']}")
        add_line()
        
        # Chi tiết từng mẫu
        add_separator("-")
        add_line("📋 PHẦN 4: CHI TIẾT TỪNG MẪU")
        add_separator("-")
        
        for result in self.results:
            status = "✅" if result['correct'] else "❌"
            add_line(f"\n{status} {result['folder']}")
            add_line(f"   Ground Truth: {result['ground_truth']}")
            add_line(f"   Prediction:   {result['prediction']} ({result['confidence']:.2%})")
            add_line(f"   Top 3: {result['top_3']}")
            
            det = result['detection_metrics']
            add_line(f"   Detection: P={det['precision']:.2f}, R={det['recall']:.2f}, F1={det['f1']:.2f} (TP={det['tp']}, FP={det['fp']}, FN={det['fn']})")
            
            # Show labels comparison
            yolo_only = set(result['yolo_labels']) - set(result['gt_labels'])
            gt_only = set(result['gt_labels']) - set(result['yolo_labels'])
            if yolo_only:
                add_line(f"   YOLO extra (FP): {list(yolo_only)[:5]}{'...' if len(yolo_only) > 5 else ''}")
            if gt_only:
                add_line(f"   GT missed (FN): {list(gt_only)[:5]}{'...' if len(gt_only) > 5 else ''}")
        
        add_line()
        add_separator()
        add_line("KẾT THÚC BÁO CÁO")
        add_separator()
        
        return "\n".join(report_lines)
    
    def save_report(self, output_dir="evaluation_results"):
        """
        Lưu báo cáo ra file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        report = self.generate_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n💾 Báo cáo đã được lưu tại: {filepath}")
        
        return filepath


def main():
    """
    Hàm main để chạy đánh giá
    """
    # Cấu hình
    FRAME_DIR = "assets/Frame"
    MODEL_PATH = "backend/weight/best.pt"
    CSV_PATH = "backend/uploads/artifacts/merged_data.csv"
    
    # Kiểm tra files tồn tại
    if not os.path.exists(FRAME_DIR):
        print(f"❌ Không tìm thấy thư mục: {FRAME_DIR}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model: {MODEL_PATH}")
        return
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ Không tìm thấy CSV: {CSV_PATH}")
        return
    
    # Khởi tạo và chạy đánh giá
    evaluator = SystemEvaluator(FRAME_DIR, MODEL_PATH, CSV_PATH)
    
    # Chạy đánh giá
    results = evaluator.run_evaluation()
    
    # In báo cáo ra console
    report = evaluator.generate_report()
    print("\n" + report)
    
    # Lưu báo cáo
    evaluator.save_report()


if __name__ == "__main__":
    main()
