"""
Đánh giá chức năng Q&A Scoring
================================
Script đánh giá hiệu quả của việc cập nhật điểm từ câu trả lời người dùng.

Flow:
1. Load ảnh từ dataset → YOLO detect → Bayesian logits ban đầu
2. Sinh câu hỏi dựa trên unsatisfied constraints
3. Test với các câu trả lời: ["có", "không", "có vẻ vậy", "hình như không"]
4. Tính điểm trước/sau cho festival ground truth
5. Đánh giá tỉ lệ câu trả lời đúng

Output: evaluation_results/qa_evaluation_YYYYMMDD_HHMMSS.txt
"""

import os
import sys
import json
import numpy as np
import cv2
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# Thêm backend vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services import YOLOCSVPipeline, ObjectDetection, BayesianFestivalClassifier
from constraintsDB import CONSTRAINTS_DB


def sigmoid(x):
    """Hàm sigmoid để chuyển logit thành probability"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


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

# ==========================================
# MAPPING TÊN VIDEO -> TÊN LỄ HỘI
# Đặt tên file video theo lễ hội, ví dụ: "Chợ Nổi.mp4", "Ngũ Âm (dễ).mp4"
# Hoặc đặt video trong folder con có tên lễ hội
# ==========================================
VIDEO_NAME_TO_FESTIVAL = {
    "Chợ Nổi": "Chợ nổi Cái Răng",
    "Cho Noi": "Chợ nổi Cái Răng",
    "Ngũ Âm": "Nhạc Ngũ Âm người Khmer",
    "Ngu Am": "Nhạc Ngũ Âm người Khmer",
    "Dù Khê": "Sân Khấu Dù Kê",
    "Du Khe": "Sân Khấu Dù Kê",
    "Nghinh Ông": "Nghinh Ông",
    "Nghinh Ong": "Nghinh Ông",
    "Kỳ Yên": "Lễ hội Kỳ Yên Đình Bình Thủy",
    "Ky Yen": "Lễ hội Kỳ Yên Đình Bình Thủy",
    "Thác Côn": "Lễ hội thác côn",
    "Thac Con": "Lễ hội thác côn",
    "Ok Bom Boc": "Ooc Bom Bóc",
    "Chol Chnam Thmay": "Tết Choi Chnam Thmay",
    "Đờn Ca Tài Tử": "Đờn ca tài tử",
    "Don Ca Tai Tu": "Đờn ca tài tử",
}

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

# Mảng câu trả lời test (8 mức độ chắc chắn)

TEST_ANSWERS = [
    # Positive answers (khẳng định)
    {"text": "chắc có", "is_positive": True, "confidence": 0.9},
    {"text": "có", "is_positive": True, "confidence": 1.0},
    {"text": "hình như có", "is_positive": True, "confidence": 0.5},
    {"text": "có lẽ có", "is_positive": True, "confidence": 0.6},
    # Negative answers (phủ định)
    {"text": "chắc không", "is_positive": False, "confidence": 0.9},
    {"text": "không", "is_positive": False, "confidence": 1.0},
    {"text": "hình như không", "is_positive": False, "confidence": 0.35},
    {"text": "có lẽ không", "is_positive": False, "confidence": 0.4},
]

# UNCERTAINTY_RULES (copy từ services.py để tính toán offline)
UNCERTAINTY_RULES = {
    # Khẳng định chắc chắn
    "có": {"status": True, "confidence": 1.0},
    "đúng": {"status": True, "confidence": 1.0},
    "đúng vậy": {"status": True, "confidence": 1.0},
    "chính xác": {"status": True, "confidence": 1.0},
    "chắc có": {"status": True, "confidence": 0.9},
    # Phủ định chắc chắn
    "không": {"status": False, "confidence": 1.0},
    "không có": {"status": False, "confidence": 1.0},
    "không thấy": {"status": False, "confidence": 1.0},
    "chắc không": {"status": False, "confidence": 0.9},
    # Khẳng định không chắc
    "có vẻ vậy": {"status": True, "confidence": 0.7},
    "có thể": {"status": True, "confidence": 0.7},
    "có lẽ": {"status": True, "confidence": 0.7},
    "có lẽ có": {"status": True, "confidence": 0.6},
    "hình như có": {"status": True, "confidence": 0.5},
    "có vẻ có": {"status": True, "confidence": 0.5},
    # Phủ định không chắc
    "có lẽ không": {"status": False, "confidence": 0.4},
    "hình như không": {"status": False, "confidence": 0.35},
    "có vẻ không": {"status": False, "confidence": 0.35},
    "không chắc": {"status": False, "confidence": 0.3},
    "không rõ": {"status": False, "confidence": 0.3},
}


class SimpleConstraintChecker:
    """Constraint checker đơn giản cho evaluation"""
    
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


class QAEvaluator:
    def __init__(self, frame_dir, model_path, csv_path, api_key=None, video_dir=None):
        """
        Khởi tạo QA Evaluator
        
        Args:
            frame_dir: Thư mục chứa các folder Frame (ảnh JPG)
            model_path: Đường dẫn model YOLO
            csv_path: Đường dẫn file CSV mapping
            api_key: API key cho LLM (optional)
            video_dir: Thư mục chứa video để đánh giá (optional)
        """
        self.frame_dir = frame_dir
        self.video_dir = video_dir
        self.api_key = api_key
        
        print("=" * 70)
        print("🚀 KHỞI TẠO HỆ THỐNG ĐÁNH GIÁ Q&A")
        print("=" * 70)
        
        # Khởi tạo YOLO pipeline
        print("\n📦 Đang load YOLO model...")
        self.yolo_pipe = YOLOCSVPipeline(model_path, csv_path)
        
        # Khởi tạo Constraint Checker
        print("📦 Đang khởi tạo Constraint Checker...")
        self.checker = SimpleConstraintChecker()
        
        # Khởi tạo Bayesian Classifier nếu có API key (để sinh câu hỏi bằng LLM)
        self.classifier = None
        if api_key:
            print("📦 Đang khởi tạo Bayesian Classifier với LLM...")
            self.classifier = BayesianFestivalClassifier(api_key=api_key)
        
        # Kết quả
        self.results = []
        
        print("\n✅ Khởi tạo hoàn tất!")
        print("=" * 70)
    
    def parse_folder_name(self, folder_name):
        """Parse tên thư mục để lấy tên lễ hội và độ khó"""
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
        """Load ground truth labels từ file JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        labels = set()
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            if label:
                labels.add(label)
        
        return labels
    
    def detect_from_image(self, image_path, frame_id=0):
        """Chạy YOLO detection trên ảnh"""
        results = self.yolo_pipe.model.predict(image_path, verbose=False)
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return []
        
        detection_groups = defaultdict(lambda: {'confidences': [], 'bboxs': []})
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            subclass = self.yolo_pipe.model.names[cls_id]
            
            detection_groups[subclass]['confidences'].append(conf)
            detection_groups[subclass]['bboxs'].append(bbox)
        
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
    
    def generate_questions_offline(self, candidates, festival_unsatisfied, max_questions=3, max_features_per_question=3):
        """
        Sinh câu hỏi offline (không dùng LLM) để đánh giá
        
        Returns:
            List[dict]: Danh sách câu hỏi với metadata
        """
        # Thu thập features và đếm số festival liên quan
        feature_to_festivals = {}
        feature_weights = {}
        
        for fest in candidates:
            rules = festival_unsatisfied.get(fest, [])
            for rule in rules:
                params = rule[1]
                weight = rule[3]
                for feature in params:
                    if feature not in feature_to_festivals:
                        feature_to_festivals[feature] = set()
                        feature_weights[feature] = 0
                    feature_to_festivals[feature].add(fest)
                    feature_weights[feature] = max(feature_weights[feature], weight)
        
        if not feature_to_festivals:
            return []
        
        # Sắp xếp features theo số festival và weight
        sorted_features = sorted(
            feature_to_festivals.keys(),
            key=lambda f: (len(feature_to_festivals[f]), feature_weights[f]),
            reverse=True
        )
        
        # Chia features thành các nhóm
        feature_groups = []
        for i in range(0, len(sorted_features), max_features_per_question):
            group = sorted_features[i:i + max_features_per_question]
            if group:
                feature_groups.append(group)
            if len(feature_groups) >= max_questions:
                break
        
        # Tạo câu hỏi
        questions = []
        for idx, group in enumerate(feature_groups):
            related_festivals = set()
            for feature in group:
                related_festivals.update(feature_to_festivals[feature])
            
            # Tạo text câu hỏi đơn giản
            if len(group) == 1:
                question_text = f"Bạn có thấy {group[0]} không?"
            else:
                feature_list = ", ".join(group[:-1]) + f" hoặc {group[-1]}"
                question_text = f"Bạn có thấy {feature_list} không?"
            
            questions.append({
                "question_id": idx + 1,
                "question_text": question_text,
                "target_features": group,
                "related_festivals": list(related_festivals),
                "feature_details": {
                    f: {
                        "festivals": list(feature_to_festivals[f]),
                        "weight": feature_weights[f]
                    } for f in group
                }
            })
        
        return questions
    
    def simulate_answer_parsing(self, answer_text, feature):
        """
        Mô phỏng việc parse câu trả lời cho MỘT feature
        
        Args:
            answer_text: Câu trả lời ("có", "không", "có vẻ vậy", "hình như không")
            feature: Tên feature
            
        Returns:
            Dict: {feature: {"status": bool, "confidence": float}}
        """
        # Tìm rule tương ứng
        answer_lower = answer_text.lower().strip()
        
        if answer_lower in UNCERTAINTY_RULES:
            rule = UNCERTAINTY_RULES[answer_lower]
        else:
            # Default: "không rõ"
            rule = {"status": False, "confidence": 0.3}
        
        # Chỉ áp dụng cho 1 feature
        return {
            feature: {
                "status": rule["status"],
                "confidence": rule["confidence"]
            }
        }
    
    def update_logits_from_answer(self, festival_logits, candidates, festival_unsatisfied, parsed_answer):
        """
        Cập nhật điểm Logit dựa trên parsed_answer
        (Copy logic từ services.py)
        """
        final_logits = festival_logits.copy()
        
        for fest in candidates:
            unsatisfied_rules = festival_unsatisfied.get(fest, [])
            
            for rule in unsatisfied_rules:
                params = rule[1]
                weight = rule[3]
                
                for param in params:
                    if param in parsed_answer:
                        data = parsed_answer[param]
                        status = data.get("status")
                        conf = data.get("confidence", 0.5)
                        
                        if status is True:
                            delta = weight * conf
                            final_logits[fest] += delta
                        elif status is False:
                            penalty = (weight * conf) / 2
                            final_logits[fest] -= penalty
        
        return final_logits
    
    def evaluate_single_folder(self, folder_path, ground_truth_festival):
        """
        Đánh giá Q&A scoring cho một folder
        """
        # Lấy tất cả file jpg
        all_files = os.listdir(folder_path)
        jpg_files = sorted([f for f in all_files if f.endswith('.jpg')])
        
        if not jpg_files:
            return None
        
        # YOLO detection trên tất cả ảnh
        all_detections = []
        all_gt_labels = set()
        
        for i, jpg_file in enumerate(jpg_files):
            jpg_path = os.path.join(folder_path, jpg_file)
            json_path = jpg_path.replace('.jpg', '.json')
            
            detections = self.detect_from_image(jpg_path, frame_id=i)
            all_detections.extend(detections)
            
            if os.path.exists(json_path):
                gt_labels = self.load_ground_truth_from_json(json_path)
                all_gt_labels.update(gt_labels)
        
        if not all_detections:
            return None
        
        # Tính logits ban đầu
        initial_logits, festival_unsatisfied, festival_satisfied = self.checker.calculate_logits(all_detections)
        
        # Xác định candidates (top festivals) kèm xác suất
        probs = {f: sigmoid(l) for f, l in initial_logits.items()}
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        candidates = [f for f, p in sorted_probs[:3]]  # Top 3
        candidate_probs = {f: float(probs[f]) for f in candidates}  # % cho từng candidate
        
        # Sinh câu hỏi
        questions = self.generate_questions_offline(candidates, festival_unsatisfied)
        
        if not questions:
            return None
        
        # Kết quả cho folder này
        folder_result = {
            "folder": os.path.basename(folder_path),
            "ground_truth": ground_truth_festival,
            "initial_logit_gt": initial_logits.get(ground_truth_festival, 0.0),
            "initial_prob_gt": float(sigmoid(initial_logits.get(ground_truth_festival, 0.0))),
            "candidates": candidates,
            "candidate_probs": candidate_probs,
            "questions": [],
            "gt_in_gt_labels": all_gt_labels,  # Ground truth labels từ JSON
        }
        
        # Test từng câu hỏi - MỖI FEATURE RIÊNG LẺ với các câu trả lời
        for question in questions:
            question_result = {
                "question_id": question["question_id"],
                "question_text": question["question_text"],
                "target_features": question["target_features"],
                "related_festivals": question["related_festivals"],
                "feature_tests": []  # Kết quả test từng feature riêng
            }
            
            # Test TỪNG FEATURE riêng lẻ
            for feature in question["target_features"]:
                feature_result = {
                    "feature": feature,
                    "in_ground_truth": feature in all_gt_labels,
                    "answers": []
                }
                
                # Test với từng câu trả lời cho feature này
                for answer_info in TEST_ANSWERS:
                    answer_text = answer_info["text"]
                    
                    # Parse answer cho 1 feature
                    parsed = self.simulate_answer_parsing(answer_text, feature)
                    
                    # Cập nhật logits
                    updated_logits = self.update_logits_from_answer(
                        initial_logits, candidates, festival_unsatisfied, parsed
                    )
                    
                    # Tính delta cho ground truth festival
                    before_logit = initial_logits.get(ground_truth_festival, 0.0)
                    after_logit = updated_logits.get(ground_truth_festival, 0.0)
                    delta = after_logit - before_logit
                    
                    before_prob = float(sigmoid(before_logit))
                    after_prob = float(sigmoid(after_logit))
                    
                    feature_result["answers"].append({
                        "answer_text": answer_text,
                        "is_positive": answer_info["is_positive"],
                        "confidence": answer_info["confidence"],
                        "before_logit": round(before_logit, 4),
                        "after_logit": round(after_logit, 4),
                        "delta": round(delta, 4),
                        "before_prob": round(before_prob, 4),
                        "after_prob": round(after_prob, 4),
                    })
                
                question_result["feature_tests"].append(feature_result)
            
            folder_result["questions"].append(question_result)
        
        return folder_result
    
    def parse_video_name(self, filename):
        """
        Parse tên file video để lấy tên lễ hội và độ khó.
        Hỗ trợ các format:
          - "Chợ Nổi.mp4" → ("Chợ nổi Cái Răng", None)
          - "Chợ Nổi (dễ).mp4" → ("Chợ nổi Cái Răng", "easy")
          - "Ngũ Âm (khó).mp4" → ("Nhạc Ngũ Âm người Khmer", "hard")
        """
        # Bỏ extension
        name_no_ext = os.path.splitext(filename)[0]
        
        # Xác định độ khó
        difficulty = None
        base_name = name_no_ext
        if "(dễ)" in name_no_ext:
            difficulty = "easy"
            base_name = name_no_ext.replace("(dễ)", "").strip()
        elif "(khó)" in name_no_ext:
            difficulty = "hard"
            base_name = name_no_ext.replace("(khó)", "").strip()
        
        # Tìm trong mapping
        festival_name = VIDEO_NAME_TO_FESTIVAL.get(base_name)
        
        # Nếu không tìm thấy, thử tìm theo substring
        if not festival_name:
            for key, value in VIDEO_NAME_TO_FESTIVAL.items():
                if key.lower() in base_name.lower() or base_name.lower() in key.lower():
                    festival_name = value
                    break
        
        return festival_name, difficulty
    
    def evaluate_single_video(self, video_path, ground_truth_festival=None):
        """
        Đánh giá Q&A scoring cho một video.
        Sử dụng YOLOCSVPipeline.process_video() để trích frame và detect.
        
        Args:
            video_path: Đường dẫn video
            ground_truth_festival: Tên lễ hội ground truth (None nếu không biết)
            
        Returns:
            dict: Kết quả đánh giá
        """
        # Dùng process_video từ YOLOCSVPipeline
        all_detections = self.yolo_pipe.process_video(
            video_path, confidence_threshold=0.5, fps_detect=1
        )
        
        if not all_detections:
            return None
        
        # Tính logits ban đầu
        initial_logits, festival_unsatisfied, festival_satisfied = self.checker.calculate_logits(all_detections)
        
        # Xác định candidates (top festivals) kèm xác suất
        probs = {f: sigmoid(l) for f, l in initial_logits.items()}
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        candidates = [f for f, p in sorted_probs[:3]]  # Top 3
        candidate_probs = {f: float(probs[f]) for f in candidates}
        
        # Sinh câu hỏi
        questions = self.generate_questions_offline(candidates, festival_unsatisfied)
        
        if not questions:
            return None
        
        # Nếu không có ground truth, dùng candidate đầu tiên để phân tích
        eval_festival = ground_truth_festival if ground_truth_festival else candidates[0]
        
        # Kết quả cho video này
        video_result = {
            "folder": os.path.basename(video_path),
            "source_type": "video",
            "ground_truth": ground_truth_festival if ground_truth_festival else "(không xác định)",
            "predicted_festival": candidates[0] if candidates else None,
            "initial_logit_gt": initial_logits.get(eval_festival, 0.0),
            "initial_prob_gt": float(sigmoid(initial_logits.get(eval_festival, 0.0))),
            "candidates": candidates,
            "candidate_probs": candidate_probs,
            "questions": [],
            "gt_in_gt_labels": set(),  # Video không có GT labels từ JSON
            "total_detections": len(all_detections),
            "unique_subclasses": list(set(d.subclass for d in all_detections)),
        }
        
        # Test từng câu hỏi
        for question in questions:
            question_result = {
                "question_id": question["question_id"],
                "question_text": question["question_text"],
                "target_features": question["target_features"],
                "related_festivals": question["related_festivals"],
                "feature_tests": []
            }
            
            for feature in question["target_features"]:
                feature_result = {
                    "feature": feature,
                    "in_ground_truth": False,  # Video không có JSON GT
                    "answers": []
                }
                
                for answer_info in TEST_ANSWERS:
                    answer_text = answer_info["text"]
                    parsed = self.simulate_answer_parsing(answer_text, feature)
                    
                    updated_logits = self.update_logits_from_answer(
                        initial_logits, candidates, festival_unsatisfied, parsed
                    )
                    
                    before_logit = initial_logits.get(eval_festival, 0.0)
                    after_logit = updated_logits.get(eval_festival, 0.0)
                    delta = after_logit - before_logit
                    
                    before_prob = float(sigmoid(before_logit))
                    after_prob = float(sigmoid(after_logit))
                    
                    feature_result["answers"].append({
                        "answer_text": answer_text,
                        "is_positive": answer_info["is_positive"],
                        "confidence": answer_info["confidence"],
                        "before_logit": round(before_logit, 4),
                        "after_logit": round(after_logit, 4),
                        "delta": round(delta, 4),
                        "before_prob": round(before_prob, 4),
                        "after_prob": round(after_prob, 4),
                    })
                
                question_result["feature_tests"].append(feature_result)
            
            video_result["questions"].append(question_result)
        
        return video_result
    
    def run_evaluation(self):
        """Chạy đánh giá trên toàn bộ dataset"""
        print("\n" + "=" * 70)
        print("🔍 BẮT ĐẦU ĐÁNH GIÁ Q&A SCORING")
        print("=" * 70)
        
        folders = [f for f in os.listdir(self.frame_dir) 
                   if os.path.isdir(os.path.join(self.frame_dir, f)) and not f.startswith('_')]
        
        print(f"\n📁 Tìm thấy {len(folders)} thư mục")
        
        for folder_name in sorted(folders):
            folder_path = os.path.join(self.frame_dir, folder_name)
            
            festival_name, difficulty = self.parse_folder_name(folder_name)
            
            if not festival_name:
                print(f"⚠️  Bỏ qua: {folder_name}")
                continue
            
            print(f"\n📂 Đang đánh giá: {folder_name}")
            print(f"   └─ Ground Truth: {festival_name}")
            
            result = self.evaluate_single_folder(folder_path, festival_name)
            
            if result:
                result["difficulty"] = difficulty
                result["source_type"] = "frame"
                self.results.append(result)
                
                # In tóm tắt
                num_questions = len(result["questions"])
                print(f"   └─ Sinh được {num_questions} câu hỏi")
                print(f"   └─ Initial prob GT: {result['initial_prob_gt']:.2%}")
        
        # === ĐÁNH GIÁ TRÊN VIDEO (nếu có) ===
        if self.video_dir and os.path.exists(self.video_dir):
            print("\n" + "=" * 70)
            print("🎬 ĐÁNH GIÁ Q&A SCORING TRÊN VIDEO")
            print("=" * 70)
            
            video_files = []
            for f in os.listdir(self.video_dir):
                ext = os.path.splitext(f)[1].lower()
                if ext in VIDEO_EXTENSIONS and not f.startswith('_'):
                    video_files.append(f)
            
            print(f"\n🎥 Tìm thấy {len(video_files)} video")
            
            for video_name in sorted(video_files):
                video_path = os.path.join(self.video_dir, video_name)
                
                festival_name, difficulty = self.parse_video_name(video_name)
                
                if not festival_name:
                    print(f"⚠️  Bỏ qua (không xác định GT): {video_name}")
                    print(f"   💡 Đổi tên video theo lễ hội, ví dụ: 'Chợ Nổi.mp4', 'Ngũ Âm (khó).mp4'")
                    continue
                
                print(f"\n🎬 Đang đánh giá video: {video_name}")
                print(f"   └─ Ground Truth: {festival_name}")
                
                result = self.evaluate_single_video(video_path, festival_name)
                
                if result:
                    result["difficulty"] = difficulty
                    self.results.append(result)
                    
                    num_questions = len(result["questions"])
                    print(f"   └─ Sinh được {num_questions} câu hỏi")
                    print(f"   └─ Initial prob GT: {result['initial_prob_gt']:.2%}")
                    print(f"   └─ Dự đoán: {result.get('predicted_festival', 'N/A')}")
        
        return self.results
    
    def calculate_metrics(self):
        """Tính các metrics tổng hợp"""
        total_questions = 0
        total_features_tested = 0
        total_answer_tests = 0
        
        # Phân tích delta
        positive_deltas = 0
        negative_deltas = 0
        zero_deltas = 0
        
        for result in self.results:
            for question in result["questions"]:
                total_questions += 1
                for feature_test in question["feature_tests"]:
                    total_features_tested += 1
                    for answer in feature_test["answers"]:
                        total_answer_tests += 1
                        if answer["delta"] > 0:
                            positive_deltas += 1
                        elif answer["delta"] < 0:
                            negative_deltas += 1
                        else:
                            zero_deltas += 1
        
        return {
            "total_folders": len(self.results),
            "total_questions": total_questions,
            "total_features_tested": total_features_tested,
            "total_answer_tests": total_answer_tests,
            "positive_deltas": positive_deltas,
            "negative_deltas": negative_deltas,
            "zero_deltas": zero_deltas,
        }
    
    def generate_report(self):
        """Tạo báo cáo chi tiết"""
        report_lines = []
        
        def add_line(text=""):
            report_lines.append(text)
        
        def add_separator(char="=", length=80):
            add_line(char * length)
        
        # Header
        add_separator()
        add_line("BÁO CÁO ĐÁNH GIÁ Q&A SCORING")
        add_line("(Đánh giá thay đổi điểm khi trả lời câu hỏi - PER FEATURE)")
        add_separator()
        add_line(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        add_line(f"Thư mục dữ liệu: {self.frame_dir}")
        add_line(f"Câu trả lời test: {[a['text'] for a in TEST_ANSWERS]}")
        add_line()
        
        # Metrics tổng hợp
        add_separator("-")
        add_line("📊 METRICS TỔNG HỢP")
        add_separator("-")
        
        metrics = self.calculate_metrics()
        add_line(f"Tổng số folders: {metrics['total_folders']}")
        add_line(f"Tổng số câu hỏi: {metrics['total_questions']}")
        add_line(f"Tổng số features được test: {metrics['total_features_tested']}")
        add_line(f"Tổng số answer tests: {metrics['total_answer_tests']}")
        add_line()
        add_line(f"📈 Phân bố delta:")
        add_line(f"   - Tăng điểm (delta > 0): {metrics['positive_deltas']}")
        add_line(f"   - Giảm điểm (delta < 0): {metrics['negative_deltas']}")
        add_line(f"   - Không đổi (delta = 0): {metrics['zero_deltas']}")
        add_line()
        
        # Chi tiết từng folder
        add_separator("-")
        add_line("📋 CHI TIẾT TỪNG MẪU")
        add_separator("-")
        
        for result in self.results:
            add_line()
            source_icon = "🎬" if result.get("source_type") == "video" else "🎭"
            add_line(f"{source_icon} {result['folder']}")
            if result.get("source_type") == "video":
                add_line(f"   Loại: Video")
                if result.get("total_detections"):
                    add_line(f"   Tổng detections: {result['total_detections']}")
                    add_line(f"   Unique subclasses: {result.get('unique_subclasses', [])}")
                if result.get("predicted_festival"):
                    add_line(f"   Dự đoán: {result['predicted_festival']}")
            add_line(f"   Ground Truth: {result['ground_truth']}")
            add_line(f"   Initial Logit: {result['initial_logit_gt']:.4f}")
            add_line(f"   Initial Prob:  {result['initial_prob_gt']:.2%}")
            
            # Hiển thị candidates kèm phần trăm
            candidate_probs = result.get("candidate_probs", {})
            add_line(f"   Candidates:")
            for i, cand in enumerate(result['candidates'], 1):
                prob = candidate_probs.get(cand, 0.0)
                gt_marker = " ★" if cand == result['ground_truth'] else ""
                add_line(f"      {i}. {cand}: {prob:.2%}{gt_marker}")
            
            for question in result["questions"]:
                add_line()
                add_line(f"   📝 Câu hỏi {question['question_id']}: {question['question_text']}")
                add_line(f"      All Features: {question['target_features']}")
                add_line()
                
                # Hiển thị từng feature riêng
                for feature_test in question["feature_tests"]:
                    feature = feature_test["feature"]
                    in_gt = "✓ (trong GT)" if feature_test["in_ground_truth"] else "✗ (không có trong GT)"
                    
                    add_line(f"      🔹 Feature: {feature} {in_gt}")
                    
                    # Bảng kết quả
                    add_line(f"         {'Câu trả lời':<20} {'Trước':<10} {'Sau':<10} {'Delta':<12}")
                    add_line(f"         {'-'*20} {'-'*10} {'-'*10} {'-'*12}")
                    
                    for answer in feature_test["answers"]:
                        delta_str = f"+{answer['delta']:.4f}" if answer["delta"] >= 0 else f"{answer['delta']:.4f}"
                        
                        add_line(f"         {answer['answer_text']:<20} "
                                f"{answer['before_logit']:<10.4f} "
                                f"{answer['after_logit']:<10.4f} "
                                f"{delta_str:<12}")
                    add_line()
        
        add_line()
        add_separator()
        add_line("KẾT THÚC BÁO CÁO")
        add_separator()
        
        return "\n".join(report_lines)
    
    def save_report(self, output_dir="evaluation_results"):
        """Lưu báo cáo ra file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qa_evaluation_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        report = self.generate_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n💾 Báo cáo đã được lưu tại: {filepath}")
        
        return filepath


def main():
    """Hàm main"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Đánh giá Q&A Scoring cho hệ thống nhận dạng lễ hội")
    parser.add_argument("--frame-dir", default="assets/Frame", help="Thư mục chứa frame ảnh")
    parser.add_argument("--video-dir", default="assets/input", help="Thư mục chứa video (mặc định: assets/input)")
    parser.add_argument("--model-path", default="backend/weight/best.pt", help="Đường dẫn model YOLO")
    parser.add_argument("--csv-path", default="backend/uploads/artifacts/merged_data.csv", help="Đường dẫn CSV mapping")
    parser.add_argument("--no-frame", action="store_true", help="Bỏ qua đánh giá trên frame (chỉ chạy video)")
    parser.add_argument("--no-video", action="store_true", help="Bỏ qua đánh giá trên video (chỉ chạy frame)")
    
    args = parser.parse_args()
    
    FRAME_DIR = args.frame_dir
    VIDEO_DIR = args.video_dir if args.video_dir else "assets/input"
    MODEL_PATH = args.model_path
    CSV_PATH = args.csv_path
    API_KEY = None
    
    # Kiểm tra
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model: {MODEL_PATH}")
        return
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ Không tìm thấy CSV: {CSV_PATH}")
        return
    
    # Xác định thư mục frame và video
    frame_dir = FRAME_DIR if not args.no_frame and os.path.exists(FRAME_DIR) else None
    video_dir = VIDEO_DIR if not args.no_video and os.path.exists(VIDEO_DIR) else None
    
    if not frame_dir and not video_dir:
        print(f"❌ Không tìm thấy dữ liệu frame ({FRAME_DIR}) hoặc video ({VIDEO_DIR})")
        return
    
    if frame_dir:
        print(f"📁 Frame dir: {frame_dir}")
    else:
        print(f"⏭️  Bỏ qua đánh giá frame")
    
    if video_dir:
        print(f"🎥 Video dir: {video_dir}")
        # Hướng dẫn đặt tên video
        video_files = [f for f in os.listdir(video_dir) 
                       if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]
        unnamed = []
        for vf in video_files:
            fn, _ = QAEvaluator.parse_video_name(None, vf)
            if not fn:
                unnamed.append(vf)
        if unnamed:
            print(f"\n⚠️  Các video sau chưa đặt tên theo lễ hội (sẽ bị bỏ qua):")
            for u in unnamed:
                print(f"   - {u}")
            print(f"   💡 Đổi tên video theo lễ hội, ví dụ:")
            for key in list(VIDEO_NAME_TO_FESTIVAL.keys())[:5]:
                print(f"      '{key}.mp4' → {VIDEO_NAME_TO_FESTIVAL[key]}")
            print()
    else:
        print(f"⏭️  Bỏ qua đánh giá video")
    
    # Khởi tạo và chạy
    evaluator = QAEvaluator(
        frame_dir=frame_dir or "assets/Frame",
        model_path=MODEL_PATH,
        csv_path=CSV_PATH,
        api_key=API_KEY,
        video_dir=video_dir
    )
    
    results = evaluator.run_evaluation()
    
    # In báo cáo
    report = evaluator.generate_report()
    print("\n" + report)
    
    # Lưu
    evaluator.save_report()


if __name__ == "__main__":
    main()
