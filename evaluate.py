"""
Script đánh giá mô hình nhận diện lễ hội
- Object Detection: mAP, Precision, Recall, F1
- Festival Classification: Accuracy, Confusion Matrix
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import YOLO và Bayesian Classifier
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from ultralytics import YOLO
from backend.services import BayesianFestivalClassifier, ObjectDetection, sigmoid
from backend.constraintsDB import CONSTRAINTS_DB
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CẤU HÌNH
# ==========================================
CONFIG = {
    "model_path": "weight/best.pt",
    "csv_path": "artifacts/merged_data.csv",
    "frame_dir": "assets/Frame",
    "iou_threshold": 0.5,
    "confidence_threshold": 0.5,
    "api_key": os.getenv("GEMINI_API_KEY")
}

# Mapping tên thư mục → tên lễ hội trong CONSTRAINTS_DB
FOLDER_TO_FESTIVAL = {
    "Frame Chợ Nổi": "Chợ nổi Cái Răng",
    "Frame Ok Bom Boc": "Ooc Bom Bóc",
    "Frame Chol Chnam Thmay": "Tết Choi Chnam Thmay",
    "Frame Dù Khê": "Sân Khấu Dù Kê",
    "Frame Kỳ Yên": "Lễ hội Kỳ Yên Đình Bình Thủy",
    "Frame Nghinh Ông": "Nghinh Ông",
    "Frame Ngũ Âm": "Nhạc Ngũ Âm người Khmer",
    "Frame Thác Côn": "Lễ hội thác côn",
    "Frame Đờn Ca Tài Tử": "Đờn ca tài tử"
}

# Mapping để normalize labels không khớp giữa JSON và YOLO
LABEL_NORMALIZE = {
    "Ao khoac vai": "Ao Khoac vai",
    "Moi doi": "Moi do",  # Có thể là typo
    "hoa sen": "Hoa sen"
}


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Tính IoU giữa 2 bounding boxes
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    
    return intersection / union


def parse_labelme_json(json_path: str) -> List[Dict]:
    """
    Parse file JSON từ LabelMe format
    Returns: List[{"label": str, "bbox": [x1, y1, x2, y2]}]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = []
    for shape in data.get('shapes', []):
        label = shape['label']
        
        # Normalize label nếu cần
        label = LABEL_NORMALIZE.get(label, label)
        
        points = shape['points']
        
        # Chuyển đổi points thành bbox [x1, y1, x2, y2]
        if len(points) == 2:
            # Format: [[x1, y1], [x2, y2]]
            x1, y1 = points[0]
            x2, y2 = points[1]
        elif len(points) == 4:
            # Format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        
        # Đảm bảo x1 < x2, y1 < y2
        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        
        annotations.append({
            "label": label,
            "bbox": bbox
        })
    
    return annotations


def get_festival_from_folder(folder_name: str) -> Optional[str]:
    """
    Lấy tên lễ hội từ tên thư mục
    """
    # Bỏ phần (dễ) hoặc (khó)
    base_name = folder_name.replace(" (dễ)", "").replace(" (khó)", "")
    return FOLDER_TO_FESTIVAL.get(base_name)


# ==========================================
# OBJECT DETECTION EVALUATION
# ==========================================

class ObjectDetectionEvaluator:
    """Đánh giá Object Detection"""
    
    def __init__(self, model_path: str, iou_threshold: float = 0.5, 
                 confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        
        # Metrics storage
        self.all_predictions = []  # [(label, confidence, is_tp)]
        self.all_ground_truths = defaultdict(int)  # {label: count}
        self.class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
    def evaluate_image(self, image_path: str, ground_truths: List[Dict]) -> Dict:
        """
        Đánh giá trên 1 ảnh
        Returns: {"predictions": [...], "matches": [...]}
        """
        # YOLO predict
        results = self.model.predict(image_path, verbose=False, conf=self.confidence_threshold)
        
        predictions = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    predictions.append({
                        "label": result.names[int(box.cls)],
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf)
                    })
        
        # Match predictions với ground truths
        gt_matched = [False] * len(ground_truths)
        pred_results = []
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_matched[gt_idx]:
                    continue
                if pred["label"].lower() != gt["label"].lower():
                    continue
                    
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            is_tp = best_iou >= self.iou_threshold and best_gt_idx >= 0
            if is_tp:
                gt_matched[best_gt_idx] = True
            
            pred_results.append({
                **pred,
                "is_tp": is_tp,
                "iou": best_iou
            })
            
            # Update metrics
            label = pred["label"]
            if is_tp:
                self.class_metrics[label]["tp"] += 1
            else:
                self.class_metrics[label]["fp"] += 1
            
            self.all_predictions.append((label, pred["confidence"], is_tp))
        
        # Count false negatives (ground truths không match)
        for gt_idx, gt in enumerate(ground_truths):
            self.all_ground_truths[gt["label"]] += 1
            if not gt_matched[gt_idx]:
                self.class_metrics[gt["label"]]["fn"] += 1
        
        return {
            "predictions": pred_results,
            "ground_truths": ground_truths,
            "matched": sum(gt_matched),
            "total_gt": len(ground_truths),
            "total_pred": len(predictions)
        }
    
    def calculate_ap(self, label: str) -> float:
        """Tính Average Precision cho 1 class"""
        # Lọc predictions của class này
        class_preds = [(conf, is_tp) for l, conf, is_tp in self.all_predictions if l == label]
        
        if not class_preds:
            return 0.0
        
        # Sort theo confidence giảm dần
        class_preds.sort(key=lambda x: x[0], reverse=True)
        
        total_gt = self.all_ground_truths[label]
        if total_gt == 0:
            return 0.0
        
        # Tính precision-recall curve
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        
        for conf, is_tp in class_preds:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_gt
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Tính AP (area under PR curve)
        # Sử dụng 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            if prec_at_recall:
                ap += max(prec_at_recall) / 11
        
        return ap
    
    def get_summary(self) -> Dict:
        """Tính toán tổng kết metrics"""
        summary = {
            "per_class": {},
            "overall": {}
        }
        
        all_aps = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for label in set(self.all_ground_truths.keys()) | set(self.class_metrics.keys()):
            metrics = self.class_metrics[label]
            tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            ap = self.calculate_ap(label)
            
            summary["per_class"][label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "ap": round(ap, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": self.all_ground_truths[label]
            }
            
            all_aps.append(ap)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        summary["overall"] = {
            "mAP@0.5": round(np.mean(all_aps) if all_aps else 0, 4),
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1": round(overall_f1, 4),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_predictions": len(self.all_predictions),
            "total_ground_truths": sum(self.all_ground_truths.values()),
            "num_classes": len(summary["per_class"])
        }
        
        return summary


# ==========================================
# FESTIVAL CLASSIFICATION EVALUATION (PER IMAGE)
# ==========================================

class FestivalClassificationEvaluator:
    """Đánh giá Festival Classification - theo từng ảnh"""
    
    def __init__(self, model_path: str, csv_path: str, api_key: str,
                 confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.csv_path = csv_path
        self.mapping_df = pd.read_csv(csv_path)
        self.mapping_df.columns = self.mapping_df.columns.str.strip()
        self.classifier = BayesianFestivalClassifier(api_key)
        self.confidence_threshold = confidence_threshold
        
        # Results storage
        self.results = []  # Kết quả từng ảnh
        self.festivals = list(CONSTRAINTS_DB.keys())
    
    def classify_single_image(self, image_path: str) -> Tuple[Optional[str], float, Dict]:
        """
        Phân loại lễ hội từ 1 ảnh đơn lẻ
        Returns: (predicted_festival, confidence, all_probs)
        """
        # YOLO predict
        results = self.model.predict(image_path, verbose=False, 
                                    conf=self.confidence_threshold)
        
        # Tạo ObjectDetection từ detections
        detections = []
        detection_by_subclass = defaultdict(list)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_name = result.names[int(box.cls)]
                    confidence = float(box.conf)
                    bbox = box.xyxy[0].tolist()
                    
                    detection_by_subclass[class_name].append({
                        "confidence": confidence,
                        "bbox": bbox
                    })
        
        # Gom thành ObjectDetection cho mỗi subclass
        for subclass, dets in detection_by_subclass.items():
            avg_conf = np.mean([d["confidence"] for d in dets])
            bboxs = [d["bbox"] for d in dets]
            
            obj = ObjectDetection(
                subclass=subclass,
                confidence=avg_conf,
                frame_id=0,
                time_stamp=0,
                count=len(dets),
                bboxs=bboxs
            )
            detections.append(obj)
        
        if not detections:
            return None, 0.0, {}
        
        # Tính Bayesian logits
        logits, unsatisfied, satisfied = self.classifier.calculate_initial_logits(detections)
        
        # Chuyển sang probabilities
        probs = {f: sigmoid(l) for f, l in logits.items()}
        
        # Lấy festival có prob cao nhất
        if probs:
            best_festival = max(probs, key=probs.get)
            best_conf = probs[best_festival]
            return best_festival, best_conf, probs
        
        return None, 0.0, {}
    
    def evaluate_image(self, image_path: str, ground_truth: str):
        """Đánh giá 1 ảnh"""
        predicted, confidence, all_probs = self.classify_single_image(image_path)
        
        self.results.append({
            "ground_truth": ground_truth,
            "predicted": predicted,
            "confidence": confidence,
            "all_probs": all_probs,
            "image": os.path.basename(image_path)
        })
        
        return predicted, confidence
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Tạo confusion matrix"""
        festival_to_idx = {f: i for i, f in enumerate(self.festivals)}
        
        n = len(self.festivals)
        cm = np.zeros((n, n), dtype=int)
        
        for result in self.results:
            gt = result["ground_truth"]
            pred = result["predicted"]
            
            if gt in festival_to_idx and pred in festival_to_idx:
                cm[festival_to_idx[gt]][festival_to_idx[pred]] += 1
        
        return cm
    
    def get_summary(self) -> Dict:
        """Tính toán tổng kết"""
        correct = sum(1 for r in self.results if r["ground_truth"] == r["predicted"])
        total = len(self.results)
        
        # Per-class accuracy
        per_class = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in self.results:
            gt = r["ground_truth"]
            per_class[gt]["total"] += 1
            if r["ground_truth"] == r["predicted"]:
                per_class[gt]["correct"] += 1
        
        per_class_acc = {}
        for festival, stats in per_class.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            per_class_acc[festival] = {
                "accuracy": round(acc, 4),
                "correct": stats["correct"],
                "total": stats["total"]
            }
        
        return {
            "overall_accuracy": round(correct / total if total > 0 else 0, 4),
            "correct": correct,
            "total": total,
            "per_class": per_class_acc,
            "confusion_matrix": self.get_confusion_matrix()
        }


# ==========================================
# VISUALIZATION
# ==========================================

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix"):
    """Vẽ confusion matrix"""
    plt.figure(figsize=(12, 10))
    
    # Normalize để hiển thị %
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return plt.gcf()


def plot_class_metrics(metrics: Dict, title: str = "Per-Class Metrics"):
    """Vẽ biểu đồ metrics theo class"""
    classes = list(metrics.keys())
    
    # Chỉ lấy top 15 classes có nhiều support nhất
    sorted_classes = sorted(classes, key=lambda x: metrics[x].get('support', 0), reverse=True)[:15]
    
    precisions = [metrics[c]['precision'] for c in sorted_classes]
    recalls = [metrics[c]['recall'] for c in sorted_classes]
    f1s = [metrics[c]['f1'] for c in sorted_classes]
    
    x = np.arange(len(sorted_classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_festival_accuracy(per_class: Dict, title: str = "Festival Classification Accuracy"):
    """Vẽ biểu đồ accuracy theo lễ hội"""
    festivals = list(per_class.keys())
    accuracies = [per_class[f]['accuracy'] for f in festivals]
    totals = [per_class[f]['total'] for f in festivals]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(festivals)))
    bars = ax.barh(festivals, accuracies, color=colors)
    
    # Thêm label
    for bar, total in zip(bars, totals):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.1%} (n={total})',
                va='center', fontsize=10)
    
    ax.set_xlabel('Accuracy')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.2)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


# ==========================================
# MAIN EVALUATION
# ==========================================

def main():
    print("=" * 70)
    print("🎯 ĐÁNH GIÁ MÔ HÌNH NHẬN DIỆN LỄ HỘI")
    print("=" * 70)
    
    frame_dir = Path(CONFIG["frame_dir"])
    
    # Lấy tất cả thư mục con (bỏ _delete_)
    folders = [f for f in frame_dir.iterdir() 
               if f.is_dir() and not f.name.startswith('_')]
    
    print(f"\n📁 Tìm thấy {len(folders)} thư mục frames")
    
    # ==========================================
    # PHẦN 1: OBJECT DETECTION EVALUATION
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 PHẦN 1: ĐÁNH GIÁ OBJECT DETECTION")
    print("=" * 70)
    
    od_evaluator = ObjectDetectionEvaluator(
        model_path=CONFIG["model_path"],
        iou_threshold=CONFIG["iou_threshold"],
        confidence_threshold=CONFIG["confidence_threshold"]
    )
    
    total_images = 0
    
    for folder in tqdm(folders, desc="Đánh giá Object Detection"):
        # Lấy tất cả json files
        json_files = list(folder.glob("*.json"))
        
        for json_file in json_files:
            # Tìm ảnh tương ứng
            image_path = json_file.with_suffix('.jpg')
            if not image_path.exists():
                image_path = json_file.with_suffix('.png')
            if not image_path.exists():
                continue
            
            # Parse ground truth
            ground_truths = parse_labelme_json(str(json_file))
            
            if ground_truths:
                od_evaluator.evaluate_image(str(image_path), ground_truths)
                total_images += 1
    
    # Lấy summary
    od_summary = od_evaluator.get_summary()
    
    print(f"\n✅ Đã đánh giá {total_images} ảnh")
    print(f"\n📈 KẾT QUẢ OBJECT DETECTION:")
    print("-" * 50)
    print(f"   mAP@0.5:     {od_summary['overall']['mAP@0.5']:.4f}")
    print(f"   Precision:   {od_summary['overall']['precision']:.4f}")
    print(f"   Recall:      {od_summary['overall']['recall']:.4f}")
    print(f"   F1-Score:    {od_summary['overall']['f1']:.4f}")
    print("-" * 50)
    print(f"   Total TP:    {od_summary['overall']['total_tp']}")
    print(f"   Total FP:    {od_summary['overall']['total_fp']}")
    print(f"   Total FN:    {od_summary['overall']['total_fn']}")
    print(f"   Num Classes: {od_summary['overall']['num_classes']}")
    
    # Top 10 classes theo F1
    print(f"\n📊 TOP 10 CLASSES (theo F1-Score):")
    print("-" * 70)
    print(f"{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    
    sorted_classes = sorted(od_summary['per_class'].items(), 
                           key=lambda x: x[1]['f1'], reverse=True)[:10]
    for cls, metrics in sorted_classes:
        print(f"{cls:<30} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['support']:>10}")
    
    # ==========================================
    # PHẦN 2: FESTIVAL CLASSIFICATION EVALUATION
    # ==========================================
    print("\n" + "=" * 70)
    print("🎭 PHẦN 2: ĐÁNH GIÁ FESTIVAL CLASSIFICATION (TỪNG ẢNH)")
    print("=" * 70)
    
    fc_evaluator = FestivalClassificationEvaluator(
        model_path=CONFIG["model_path"],
        csv_path=CONFIG["csv_path"],
        api_key=CONFIG["api_key"],
        confidence_threshold=CONFIG["confidence_threshold"]
    )
    
    # Đánh giá từng ảnh trong tất cả folders
    total_images_fc = 0
    for folder in tqdm(folders, desc="Đánh giá Festival Classification"):
        ground_truth = get_festival_from_folder(folder.name)
        
        if ground_truth:
            # Lấy tất cả ảnh trong folder
            image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            
            for image_path in image_files:
                fc_evaluator.evaluate_image(str(image_path), ground_truth)
                total_images_fc += 1
    
    # Lấy summary
    fc_summary = fc_evaluator.get_summary()
    
    print(f"\n✅ Đã đánh giá {fc_summary['total']} ảnh")
    print(f"\n📈 KẾT QUẢ FESTIVAL CLASSIFICATION:")
    print("-" * 50)
    print(f"   Overall Accuracy: {fc_summary['overall_accuracy']:.4f} ({fc_summary['correct']}/{fc_summary['total']})")
    print("-" * 50)
    
    print(f"\n📊 ACCURACY THEO LỄ HỘI:")
    print("-" * 70)
    print(f"{'Lễ hội':<40} {'Accuracy':>12} {'Correct':>10} {'Total':>10}")
    print("-" * 70)
    
    for festival, stats in sorted(fc_summary['per_class'].items(), 
                                  key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{festival:<40} {stats['accuracy']:>12.2%} {stats['correct']:>10} {stats['total']:>10}")
    
    # ==========================================
    # VISUALIZATION
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 TẠO BIỂU ĐỒ TRỰC QUAN")
    print("=" * 70)
    
    # Tạo thư mục output
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Confusion Matrix
    print("   📈 Vẽ Confusion Matrix...")
    cm_fig = plot_confusion_matrix(
        fc_summary['confusion_matrix'],
        fc_evaluator.festivals,
        "Festival Classification - Confusion Matrix"
    )
    cm_fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class metrics (Object Detection)
    print("   📈 Vẽ Object Detection Metrics...")
    od_fig = plot_class_metrics(
        od_summary['per_class'],
        "Object Detection - Per-Class Metrics (Top 15)"
    )
    od_fig.savefig(output_dir / "object_detection_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Festival Accuracy
    print("   📈 Vẽ Festival Classification Accuracy...")
    fc_fig = plot_festival_accuracy(
        fc_summary['per_class'],
        "Festival Classification - Accuracy by Festival"
    )
    fc_fig.savefig(output_dir / "festival_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Đã lưu biểu đồ vào thư mục: {output_dir}/")
    
    # ==========================================
    # TỔNG KẾT
    # ==========================================
    print("\n" + "=" * 70)
    print("📋 TỔNG KẾT ĐÁNH GIÁ")
    print("=" * 70)
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    OBJECT DETECTION                                  │
├─────────────────────────────────────────────────────────────────────┤
│  mAP@0.5:    {od_summary['overall']['mAP@0.5']:<10.4f}                                      │
│  Precision:  {od_summary['overall']['precision']:<10.4f}                                      │
│  Recall:     {od_summary['overall']['recall']:<10.4f}                                      │
│  F1-Score:   {od_summary['overall']['f1']:<10.4f}                                      │
├─────────────────────────────────────────────────────────────────────┤
│                   FESTIVAL CLASSIFICATION                            │
├─────────────────────────────────────────────────────────────────────┤
│  Accuracy:   {fc_summary['overall_accuracy']:<10.4f} ({fc_summary['correct']}/{fc_summary['total']} folders)                        │
└─────────────────────────────────────────────────────────────────────┘
    """)
    
    # Hiển thị biểu đồ
    print("\n🖼️  Mở biểu đồ để xem...")
    
    # Mở tất cả biểu đồ
    for img_file in output_dir.glob("*.png"):
        os.system(f"open '{img_file}'")  # macOS
    
    return od_summary, fc_summary


if __name__ == "__main__":
    main()
