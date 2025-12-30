"""
Manual Test Tool cho Chatbot Multi-turn Questions.

Cách sử dụng:
    python tools/manual_test.py --image path/to/image.jpg
    python tools/manual_test.py --folder path/to/frames/

Chế độ:
    1. Interactive: Nhập câu trả lời thực
    2. Auto: Simulate câu trả lời từ ground truth JSON
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

# Thêm path để import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from constraintsDB import CONSTRAINTS_DB, SUBCLASS_TO_FESTIVAL


def print_header():
    """In header công cụ."""
    print("\n" + "=" * 70)
    print("🎯 MANUAL TEST TOOL - CHATBOT MULTI-TURN QUESTIONS")
    print("=" * 70)


def print_section(title):
    """In tiêu đề section."""
    print(f"\n{'─' * 50}")
    print(f"📌 {title}")
    print("─" * 50)


def load_classifier():
    """Load BayesianFestivalClassifier."""
    try:
        from services import BayesianFestivalClassifier
        
        model_path = os.path.join(os.path.dirname(__file__), '..', 'weight', 'best.pt')
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'merged_data.csv')
        
        if not os.path.exists(model_path):
            print(f"❌ Không tìm thấy model: {model_path}")
            return None
        if not os.path.exists(csv_path):
            print(f"❌ Không tìm thấy CSV: {csv_path}")
            return None
            
        print(f"📦 Loading model từ: {model_path}")
        classifier = BayesianFestivalClassifier(model_path, csv_path)
        print("✅ Đã load classifier thành công!")
        return classifier
        
    except Exception as e:
        print(f"❌ Lỗi load classifier: {e}")
        return None


def get_ground_truth_from_folder(folder_name):
    """Lấy ground truth festival từ tên folder."""
    # Mapping tên folder → festival
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
    
    for key, festival in FOLDER_TO_FESTIVAL.items():
        if key.lower() in folder_name.lower():
            return festival
    return None


def simulate_answer_from_json(json_path, target_features):
    """
    Simulate câu trả lời dựa trên ground truth JSON.
    
    Args:
        json_path: Path tới file JSON annotation
        target_features: List các features cần kiểm tra
        
    Returns:
        dict: {feature: {"status": bool, "confidence": float}}
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Lấy tất cả labels từ JSON
        labels_in_json = set()
        for shape in data.get('shapes', []):
            labels_in_json.add(shape.get('label', '').strip())
        
        # Kiểm tra từng feature
        result = {}
        for feature in target_features:
            # Tìm kiếm case-insensitive
            found = any(
                feature.lower() == label.lower() or 
                feature.lower() in label.lower() or
                label.lower() in feature.lower()
                for label in labels_in_json
            )
            
            result[feature] = {
                "status": found,
                "confidence": 1.0 if found else 0.9  # Giả lập confidence
            }
        
        return result
        
    except Exception as e:
        print(f"⚠️ Lỗi đọc JSON {json_path}: {e}")
        return {}


def interactive_answer(question, target_features):
    """
    Thu thập câu trả lời tương tác từ user.
    
    Returns:
        str: Câu trả lời của user
    """
    print(f"\n❓ {question}")
    print(f"   📋 Các features cần xác nhận: {', '.join(target_features)}")
    print("\n   💡 Gợi ý trả lời:")
    print("      - 'có' / 'không' / 'có lẽ' / 'hình như'")
    print("      - 'có A nhưng không có B'")
    print("      - 'chắc chắn có A, không thấy B'")
    
    answer = input("\n👉 Trả lời: ").strip()
    return answer


def run_manual_test(image_path=None, folder_path=None, auto_mode=False):
    """
    Chạy manual test.
    
    Args:
        image_path: Path tới một ảnh
        folder_path: Path tới folder chứa frames
        auto_mode: True = simulate câu trả lời, False = nhập thủ công
    """
    print_header()
    
    # Load classifier
    classifier = load_classifier()
    if not classifier:
        return
    
    # Xác định input
    if folder_path:
        folder = Path(folder_path)
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if not image_files:
            print(f"❌ Không tìm thấy ảnh trong: {folder_path}")
            return
        
        # Lấy ảnh đầu tiên để test
        test_image = str(image_files[0])
        json_files = list(folder.glob("*.json"))
        test_json = str(json_files[0]) if json_files else None
        ground_truth = get_ground_truth_from_folder(folder.name)
        
        print(f"📁 Folder: {folder.name}")
        print(f"📷 Số ảnh: {len(image_files)}")
        print(f"🎯 Ground Truth: {ground_truth or 'Không xác định'}")
        
    elif image_path:
        test_image = image_path
        test_json = image_path.replace('.jpg', '.json').replace('.png', '.json')
        if not os.path.exists(test_json):
            test_json = None
        ground_truth = None
        
        print(f"📷 Image: {test_image}")
    else:
        print("❌ Cần cung cấp --image hoặc --folder")
        return
    
    # =========== BƯỚC 1: YOLO Detection ===========
    print_section("BƯỚC 1: YOLO Object Detection")
    
    detected_objects = classifier.pipeline.predict_and_map(test_image, show_image=False)
    
    print(f"🔍 Detected {len(detected_objects)} objects:")
    for obj in detected_objects[:10]:  # Hiển thị tối đa 10
        if obj.get('mapped_subclass'):
            print(f"   • {obj['mapped_subclass']} ({obj['confidence']:.2f})")
    if len(detected_objects) > 10:
        print(f"   ... và {len(detected_objects) - 10} objects khác")
    
    # =========== BƯỚC 2: Tính Logits ===========
    print_section("BƯỚC 2: Calculate Initial Logits")
    
    logits, unsatisfied, satisfied = classifier.calculate_initial_logits(detected_objects)
    candidates = classifier.select_candidates(logits)
    
    from services import sigmoid
    probs = {f: sigmoid(l) for f, l in logits.items()}
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📊 Top 5 Festivals:")
    for i, (fest, prob) in enumerate(sorted_probs[:5], 1):
        marker = "🔥" if fest in candidates else "  "
        print(f"   {marker} {i}. {fest}: {prob:.2%}")
    
    print(f"\n🎯 Candidates: {candidates}")
    
    # =========== BƯỚC 3: Sinh câu hỏi ===========
    print_section("BƯỚC 3: Generate Multi-turn Questions")
    
    questions = classifier.generate_multi_turn_questions(candidates, unsatisfied)
    
    print(f"📝 Sinh được {len(questions)} câu hỏi:")
    for q in questions:
        print(f"\n   Q{q['question_id']} [{q['priority']}]:")
        print(f"   Features: {', '.join(q['target_features'][:5])}...")
        print(f"   Related: {', '.join(q['related_festivals'])}")
    
    # =========== BƯỚC 4: Multi-turn Q&A ===========
    print_section("BƯỚC 4: Multi-turn Q&A Session")
    
    current_logits = logits.copy()
    qa_history = []
    
    for q in questions:
        print(f"\n{'─' * 40}")
        print(f"🔄 LƯỢT {q['question_id']}/{len(questions)} [{q['priority']}]")
        
        if auto_mode and test_json:
            # Simulate câu trả lời
            parsed_answer = simulate_answer_from_json(test_json, q['target_features'])
            
            # Tạo câu trả lời text
            yes_features = [f for f, d in parsed_answer.items() if d['status']]
            no_features = [f for f, d in parsed_answer.items() if not d['status']]
            
            if yes_features and no_features:
                answer_text = f"Có {', '.join(yes_features[:3])}. Không có {', '.join(no_features[:3])}."
            elif yes_features:
                answer_text = f"Có {', '.join(yes_features[:5])}."
            else:
                answer_text = f"Không thấy các đặc trưng này."
            
            print(f"\n❓ {q['question_text'][:100]}...")
            print(f"🤖 [AUTO] Trả lời: {answer_text}")
            
        else:
            # Interactive mode
            answer_text = interactive_answer(q['question_text'], q['target_features'])
            parsed_answer = None  # Sẽ cần LLM để parse
        
        # Cập nhật logits (nếu có parsed_answer)
        if parsed_answer:
            current_logits = classifier.update_logits_from_consolidated_answer(
                current_logits, candidates, unsatisfied, parsed_answer
            )
        
        qa_history.append({
            "turn": q['question_id'],
            "question": q['question_text'],
            "answer": answer_text
        })
        
        # Kiểm tra có cần tiếp tục không
        if not classifier.should_continue_asking(current_logits):
            print("\n✅ Đủ tự tin để kết luận, dừng hỏi sớm!")
            break
    
    # =========== BƯỚC 5: Kết luận ===========
    print_section("BƯỚC 5: Final Result")
    
    winners, final_probs = classifier.decide_final_result(current_logits)
    
    sorted_final = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📊 Kết quả cuối cùng:")
    for i, (fest, prob) in enumerate(sorted_final[:5], 1):
        marker = "🏆" if fest in winners else "  "
        print(f"   {marker} {i}. {fest}: {prob:.2%}")
    
    predicted = winners[0] if winners else sorted_final[0][0]
    
    print(f"\n🎯 Dự đoán: {predicted}")
    
    if ground_truth:
        is_correct = predicted == ground_truth
        print(f"📍 Ground Truth: {ground_truth}")
        print(f"{'✅ ĐÚNG!' if is_correct else '❌ SAI!'}")
    
    # =========== Summary ===========
    print_section("SUMMARY")
    print(f"   📷 Input: {test_image}")
    print(f"   🔍 Objects detected: {len(detected_objects)}")
    print(f"   ❓ Questions asked: {len(qa_history)}")
    print(f"   🎯 Prediction: {predicted}")
    if ground_truth:
        print(f"   📍 Ground Truth: {ground_truth}")
        print(f"   ✅ Correct: {predicted == ground_truth}")


def main():
    parser = argparse.ArgumentParser(description="Manual Test Tool for Chatbot")
    parser.add_argument("--image", "-i", help="Path to test image")
    parser.add_argument("--folder", "-f", help="Path to folder containing frames")
    parser.add_argument("--auto", "-a", action="store_true", 
                       help="Auto mode: simulate answers from JSON ground truth")
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        # Demo mode
        print("💡 Sử dụng: python manual_test.py --folder 'assets/Frame/Frame Chợ Nổi (dễ)' --auto")
        print("💡 Hoặc:   python manual_test.py --image 'path/to/image.jpg'")
        return
    
    run_manual_test(
        image_path=args.image,
        folder_path=args.folder,
        auto_mode=args.auto
    )


if __name__ == "__main__":
    main()
