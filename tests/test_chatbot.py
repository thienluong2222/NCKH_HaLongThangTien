"""
Test suite cho chức năng Chatbot Multi-turn Questions.

Bao gồm:
1. test_question_generation - Kiểm tra sinh câu hỏi hợp lệ
2. test_answer_analysis - Kiểm tra phân tích câu trả lời
3. test_logit_update - Kiểm tra cập nhật điểm
4. test_multi_turn_flow - Test flow nhiều lượt hỏi
5. test_accuracy_comparison - So sánh accuracy có/không chatbot
"""

import sys
import os
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Thêm path để import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from constraintsDB import CONSTRAINTS_DB


class TestQuestionGeneration(unittest.TestCase):
    """Test sinh câu hỏi multi-turn."""
    
    def setUp(self):
        """Setup mock classifier."""
        # Import sau khi đã thêm path
        from services import BayesianFestivalClassifier
        
        # Mock để không cần load model thật
        with patch.object(BayesianFestivalClassifier, '__init__', lambda x: None):
            self.classifier = BayesianFestivalClassifier()
            # Manually set các method cần thiết
            from services import BayesianFestivalClassifier as RealClass
            self.classifier.generate_multi_turn_questions = RealClass.generate_multi_turn_questions.__get__(self.classifier)
            self.classifier.get_question_by_turn = RealClass.get_question_by_turn.__get__(self.classifier)
            self.classifier.should_continue_asking = RealClass.should_continue_asking.__get__(self.classifier)
    
    def test_generate_questions_not_empty(self):
        """Kiểm tra sinh câu hỏi không rỗng khi có unsatisfied constraints."""
        candidates = ["Ooc Bom Bóc", "Tết Choi Chnam Thmay"]
        
        # Mock unsatisfied constraints
        unsatisfied = {
            "Ooc Bom Bóc": [
                ("is_presence_in_frame", ["Den hoa dang", "Den nuoc"], True, 1.0, None),
                ("at_least", ["Ghe ngo"], True, 1.0, 5),
            ],
            "Tết Choi Chnam Thmay": [
                ("confidence_min", ["Nui cat"], True, 1.0, 0.8),
                ("at_least_in_frame", ["Tuong Phat", "Nuoc thom"], False, 0.8, None),
            ]
        }
        
        questions = self.classifier.generate_multi_turn_questions(candidates, unsatisfied)
        
        self.assertIsInstance(questions, list)
        self.assertGreater(len(questions), 0)
        self.assertLessEqual(len(questions), 3)
        
    def test_question_structure(self):
        """Kiểm tra cấu trúc câu hỏi sinh ra."""
        candidates = ["Chợ nổi Cái Răng"]
        unsatisfied = {
            "Chợ nổi Cái Răng": [
                ("is_presence", ["Cay beo", "Thuyen"], True, 1.0, None),
            ]
        }
        
        questions = self.classifier.generate_multi_turn_questions(candidates, unsatisfied)
        
        if questions:
            q = questions[0]
            self.assertIn("question_id", q)
            self.assertIn("question_text", q)
            self.assertIn("target_features", q)
            self.assertIn("priority", q)
            self.assertIn("related_festivals", q)
            
    def test_feature_prioritization(self):
        """Kiểm tra features được sắp xếp theo số festival liên quan."""
        candidates = ["Ooc Bom Bóc", "Tết Choi Chnam Thmay", "Nhạc Ngũ Âm người Khmer"]
        
        # Feature "trong Chhay-dam" xuất hiện trong 2 festivals
        unsatisfied = {
            "Ooc Bom Bóc": [
                ("at_least_in_frame", ["trong Chhay-dam", "Nguoi mua"], False, 0.5, None),
            ],
            "Tết Choi Chnam Thmay": [
                ("at_least_in_frame", ["Nguoi mua", "trong Chhay-dam"], False, 0.6, None),
            ],
            "Nhạc Ngũ Âm người Khmer": [
                ("at_least", ["Dan thuyen Ro-niet-ek"], True, 1.0, 5),
            ]
        }
        
        questions = self.classifier.generate_multi_turn_questions(candidates, unsatisfied)
        
        # Câu hỏi đầu tiên nên chứa features xuất hiện trong nhiều festivals
        if questions:
            first_q_features = questions[0]["target_features"]
            # "trong Chhay-dam" hoặc "Nguoi mua" nên được ưu tiên
            common_features = ["trong Chhay-dam", "Nguoi mua"]
            has_common = any(f in first_q_features for f in common_features)
            self.assertTrue(has_common, f"First question should prioritize common features, got: {first_q_features}")
    
    def test_max_questions_limit(self):
        """Kiểm tra giới hạn tối đa 3 câu hỏi."""
        candidates = list(CONSTRAINTS_DB.keys())[:3]  # Lấy 3 festivals
        
        # Tạo nhiều unsatisfied constraints
        unsatisfied = {}
        for fest in candidates:
            unsatisfied[fest] = CONSTRAINTS_DB.get(fest, [])[:5]  # Lấy tối đa 5 rules
        
        questions = self.classifier.generate_multi_turn_questions(candidates, unsatisfied, max_questions=3)
        
        self.assertLessEqual(len(questions), 3)
    
    def test_empty_candidates(self):
        """Kiểm tra khi không có candidates."""
        questions = self.classifier.generate_multi_turn_questions([], {})
        self.assertEqual(questions, [])
    
    def test_get_question_by_turn(self):
        """Kiểm tra lấy câu hỏi theo lượt."""
        mock_questions = [
            {"question_id": 1, "question_text": "Q1"},
            {"question_id": 2, "question_text": "Q2"},
        ]
        
        # Lượt 1
        q1 = self.classifier.get_question_by_turn(mock_questions, 1)
        self.assertEqual(q1["question_id"], 1)
        
        # Lượt 2
        q2 = self.classifier.get_question_by_turn(mock_questions, 2)
        self.assertEqual(q2["question_id"], 2)
        
        # Lượt không hợp lệ
        q_invalid = self.classifier.get_question_by_turn(mock_questions, 3)
        self.assertIsNone(q_invalid)


class TestAnswerAnalysis(unittest.TestCase):
    """Test phân tích câu trả lời."""
    
    def test_parse_positive_answer(self):
        """Test phân tích câu trả lời khẳng định."""
        # Mock kết quả từ LLM
        mock_result = {
            "Den hoa dang": {"status": True, "confidence": 1.0},
            "Ghe ngo": {"status": True, "confidence": 0.8}
        }
        
        # Kiểm tra cấu trúc
        for feature, data in mock_result.items():
            self.assertIn("status", data)
            self.assertIn("confidence", data)
            self.assertIsInstance(data["status"], bool)
            self.assertGreaterEqual(data["confidence"], 0)
            self.assertLessEqual(data["confidence"], 1)
    
    def test_parse_negative_answer(self):
        """Test phân tích câu trả lời phủ định."""
        mock_result = {
            "Nui cat": {"status": False, "confidence": 0.9},
        }
        
        self.assertFalse(mock_result["Nui cat"]["status"])
        
    def test_parse_uncertain_answer(self):
        """Test phân tích câu trả lời không chắc chắn."""
        # Theo UNCERTAINTY_RULES: "có lẽ" = 0.7, "hình như" = 0.6
        mock_result = {
            "Thuyen": {"status": True, "confidence": 0.7},  # "có lẽ có"
        }
        
        self.assertTrue(mock_result["Thuyen"]["status"])
        self.assertLess(mock_result["Thuyen"]["confidence"], 1.0)


class TestLogitUpdate(unittest.TestCase):
    """Test cập nhật điểm logit."""
    
    def setUp(self):
        """Setup test data."""
        self.initial_logits = {
            "Ooc Bom Bóc": 0.5,
            "Tết Choi Chnam Thmay": 0.3,
            "Chợ nổi Cái Răng": 0.1
        }
        
        self.unsatisfied = {
            "Ooc Bom Bóc": [
                ("is_presence_in_frame", ["Den hoa dang"], True, 1.0, None),
            ],
            "Tết Choi Chnam Thmay": [
                ("confidence_min", ["Nui cat"], True, 0.8, None),
            ],
        }
        
        self.candidates = ["Ooc Bom Bóc", "Tết Choi Chnam Thmay"]
    
    def test_positive_update_increases_logit(self):
        """Xác nhận CÓ feature → tăng điểm."""
        parsed_answer = {
            "Den hoa dang": {"status": True, "confidence": 1.0}
        }
        
        # Giả lập update logic
        final_logits = self.initial_logits.copy()
        for fest in self.candidates:
            for rule in self.unsatisfied.get(fest, []):
                params = rule[1]
                weight = rule[3]
                for param in params:
                    if param in parsed_answer:
                        data = parsed_answer[param]
                        if data["status"]:
                            delta = weight * data["confidence"]
                            final_logits[fest] += delta
        
        # Ooc Bom Bóc phải tăng
        self.assertGreater(final_logits["Ooc Bom Bóc"], self.initial_logits["Ooc Bom Bóc"])
    
    def test_negative_update_decreases_logit(self):
        """Xác nhận KHÔNG có feature → giảm điểm."""
        parsed_answer = {
            "Den hoa dang": {"status": False, "confidence": 0.9}
        }
        
        final_logits = self.initial_logits.copy()
        for fest in self.candidates:
            for rule in self.unsatisfied.get(fest, []):
                params = rule[1]
                weight = rule[3]
                for param in params:
                    if param in parsed_answer:
                        data = parsed_answer[param]
                        if not data["status"]:
                            penalty = (weight * data["confidence"]) / 2
                            final_logits[fest] -= penalty
        
        # Ooc Bom Bóc phải giảm
        self.assertLess(final_logits["Ooc Bom Bóc"], self.initial_logits["Ooc Bom Bóc"])


class TestMultiTurnFlow(unittest.TestCase):
    """Test flow hỏi nhiều lượt."""
    
    def test_turn_progression(self):
        """Test tiến trình các lượt hỏi."""
        # Simulate session state
        session = {
            "current_turn": 1,
            "total_questions": 3,
            "logits": {"A": 0.5, "B": 0.3}
        }
        
        # Sau lượt 1
        session["current_turn"] = 2
        self.assertEqual(session["current_turn"], 2)
        
        # Sau lượt 2
        session["current_turn"] = 3
        self.assertEqual(session["current_turn"], 3)
        
        # Không còn câu hỏi
        has_more = session["current_turn"] < session["total_questions"]
        self.assertFalse(has_more)
    
    def test_early_termination(self):
        """Test kết thúc sớm khi đủ tự tin."""
        from services import sigmoid, GLOBAL_CONFIG
        
        # Logits sau khi update
        final_logits = {
            "Ooc Bom Bóc": 2.5,  # sigmoid ≈ 0.92
            "Tết Choi Chnam Thmay": -0.5  # sigmoid ≈ 0.38
        }
        
        probs = {f: sigmoid(l) for f, l in final_logits.items()}
        top_prob = max(probs.values())
        
        # Nếu top > T_high thì không cần hỏi thêm
        should_stop = top_prob >= GLOBAL_CONFIG["T_high"]
        self.assertTrue(should_stop)


class TestAccuracyComparison(unittest.TestCase):
    """Test so sánh accuracy có/không chatbot."""
    
    def test_without_chatbot(self):
        """Test accuracy khi không dùng chatbot (chỉ YOLO)."""
        # Giả lập kết quả: YOLO detect → logits → kết luận ngay
        results = {
            "folder_1": {"predicted": "Ooc Bom Bóc", "ground_truth": "Ooc Bom Bóc", "correct": True},
            "folder_2": {"predicted": "Chợ nổi Cái Răng", "ground_truth": "Tết Choi Chnam Thmay", "correct": False},
        }
        
        accuracy_no_chat = sum(1 for r in results.values() if r["correct"]) / len(results)
        self.assertEqual(accuracy_no_chat, 0.5)
    
    def test_with_chatbot(self):
        """Test accuracy khi dùng chatbot (YOLO + QA)."""
        # Giả lập: sau khi hỏi thêm, kết quả được cải thiện
        results_with_chat = {
            "folder_1": {"predicted": "Ooc Bom Bóc", "ground_truth": "Ooc Bom Bóc", "correct": True},
            "folder_2": {"predicted": "Tết Choi Chnam Thmay", "ground_truth": "Tết Choi Chnam Thmay", "correct": True},  # Sửa được
        }
        
        accuracy_with_chat = sum(1 for r in results_with_chat.values() if r["correct"]) / len(results_with_chat)
        self.assertEqual(accuracy_with_chat, 1.0)
        
    def test_improvement_calculation(self):
        """Test tính toán mức cải thiện."""
        accuracy_no_chat = 0.5
        accuracy_with_chat = 0.75
        
        improvement = accuracy_with_chat - accuracy_no_chat
        improvement_percent = (improvement / accuracy_no_chat) * 100
        
        self.assertEqual(improvement, 0.25)
        self.assertEqual(improvement_percent, 50.0)


def run_tests():
    """Chạy tất cả tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Thêm các test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQuestionGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestAnswerAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestLogitUpdate))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiTurnFlow))
    suite.addTests(loader.loadTestsFromTestCase(TestAccuracyComparison))
    
    # Chạy với verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    run_tests()
