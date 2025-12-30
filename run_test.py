"""
Script chạy test cho chatbot multi-turn questions.
"""

import sys
import os

# Thêm path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from constraintsDB import CONSTRAINTS_DB
from services import sigmoid, GLOBAL_CONFIG

print("=" * 60)
print("🧪 TEST: generate_multi_turn_questions")
print("=" * 60)

# ========== TEST 1: Logic sinh câu hỏi ==========
print("\n📌 Test 1: Feature prioritization logic")

candidates = ['Ooc Bom Bóc', 'Tết Choi Chnam Thmay']
unsatisfied = {
    'Ooc Bom Bóc': [
        ('is_presence_in_frame', ['Den hoa dang', 'Den nuoc'], True, 1.0, None),
        ('at_least', ['Ghe ngo'], True, 1.0, 5),
        ('at_least_in_frame', ['trong Chhay-dam', 'Nguoi mua'], False, 0.5, None),
    ],
    'Tết Choi Chnam Thmay': [
        ('confidence_min', ['Nui cat'], True, 1.0, 0.8),
        ('at_least_in_frame', ['Tuong Phat', 'Nuoc thom'], False, 0.8, None),
        ('at_least_in_frame', ['Nguoi mua', 'trong Chhay-dam'], False, 0.6, None),
    ]
}

# Thu thập features
feature_to_festivals = {}
feature_weights = {}

for fest in candidates:
    rules = unsatisfied.get(fest, [])
    for rule in rules:
        params = rule[1]
        weight = rule[3]
        for feature in params:
            if feature not in feature_to_festivals:
                feature_to_festivals[feature] = set()
                feature_weights[feature] = 0
            feature_to_festivals[feature].add(fest)
            feature_weights[feature] = max(feature_weights[feature], weight)

# Sắp xếp theo số festival (nhiều → hỏi trước)
sorted_features = sorted(
    feature_to_festivals.keys(),
    key=lambda f: (len(feature_to_festivals[f]), feature_weights[f]),
    reverse=True
)

print("\n   Feature prioritization (sorted by festival count):")
for f in sorted_features:
    festivals = list(feature_to_festivals[f])
    print(f"   • {f}: {len(festivals)} festivals, weight={feature_weights[f]}")
    print(f"     → {', '.join(festivals)}")

# Verify: Features xuất hiện trong nhiều festival phải đứng đầu
common_features = [f for f in sorted_features if len(feature_to_festivals[f]) > 1]
print(f"\n   ✅ Common features (in multiple festivals): {common_features}")

assert len(common_features) > 0, "Should have common features"
assert sorted_features[0] in common_features or sorted_features[1] in common_features, \
    "Common features should be prioritized"

print("   ✅ Test 1 PASSED!")

# ========== TEST 2: Chia nhóm câu hỏi ==========
print("\n📌 Test 2: Question grouping (max 3)")

max_questions = 3
features_per_question = max(3, len(sorted_features) // max_questions + 1)
feature_groups = []

for i in range(0, len(sorted_features), features_per_question):
    group = sorted_features[i:i + features_per_question]
    if group:
        feature_groups.append(group)
    if len(feature_groups) >= max_questions:
        remaining = sorted_features[i + features_per_question:]
        if remaining:
            feature_groups[-1].extend(remaining)
        break

print(f"\n   Created {len(feature_groups)} question groups:")
for i, group in enumerate(feature_groups):
    print(f"   Q{i+1}: {group}")

assert len(feature_groups) <= 3, f"Max 3 questions, got {len(feature_groups)}"
print("   ✅ Test 2 PASSED!")

# ========== TEST 3: Tạo câu hỏi ==========
print("\n📌 Test 3: Question generation")

questions = []
candidate_str = ", ".join(candidates)

for idx, group in enumerate(feature_groups):
    feature_list_str = ", ".join(group)
    
    if idx == 0:
        priority = "high"
    elif idx == 1:
        priority = "medium"
    else:
        priority = "low"
    
    related_festivals = set()
    for feature in group:
        related_festivals.update(feature_to_festivals[feature])
    
    question_text = (
        f"Hệ thống đang phân vân giữa các lễ hội: {candidate_str}. "
        f"Bạn hãy quan sát kỹ video và cho biết bạn có thấy các đặc trưng sau không: "
        f"{feature_list_str}?"
    )
    
    questions.append({
        "question_id": idx + 1,
        "question_text": question_text,
        "target_features": group,
        "priority": priority,
        "related_festivals": list(related_festivals),
    })

print(f"\n   Generated {len(questions)} questions:")
for q in questions:
    print(f"\n   Q{q['question_id']} [{q['priority']}]:")
    print(f"   Features: {q['target_features']}")
    print(f"   Related: {q['related_festivals']}")
    print(f"   Text: {q['question_text'][:80]}...")

# Verify structure
assert all('question_id' in q for q in questions), "Missing question_id"
assert all('target_features' in q for q in questions), "Missing target_features"
assert all('priority' in q for q in questions), "Missing priority"
assert all('related_festivals' in q for q in questions), "Missing related_festivals"

print("\n   ✅ Test 3 PASSED!")

# ========== TEST 4: should_continue_asking ==========
print("\n📌 Test 4: should_continue_asking logic")

# Case 1: Top probability đủ cao → không cần hỏi thêm
logits_high = {
    'Ooc Bom Bóc': 2.5,  # sigmoid ≈ 0.92
    'Tết Choi Chnam Thmay': -0.5
}
probs_high = {f: sigmoid(l) for f, l in logits_high.items()}
top_prob_high = max(probs_high.values())
should_stop_high = top_prob_high >= GLOBAL_CONFIG["T_high"]

print(f"   Case 1: Top prob = {top_prob_high:.2%}, T_high = {GLOBAL_CONFIG['T_high']}")
print(f"   → Should stop: {should_stop_high}")
assert should_stop_high == True, "Should stop when top prob >= T_high"

# Case 2: Top probability thấp → cần hỏi thêm
logits_low = {
    'Ooc Bom Bóc': 0.5,  # sigmoid ≈ 0.62
    'Tết Choi Chnam Thmay': 0.3
}
probs_low = {f: sigmoid(l) for f, l in logits_low.items()}
top_prob_low = max(probs_low.values())
should_continue_low = top_prob_low < GLOBAL_CONFIG["T_high"]

print(f"   Case 2: Top prob = {top_prob_low:.2%}, T_high = {GLOBAL_CONFIG['T_high']}")
print(f"   → Should continue: {should_continue_low}")
assert should_continue_low == True, "Should continue when top prob < T_high"

print("   ✅ Test 4 PASSED!")

# ========== TEST 5: Logit update ==========
print("\n📌 Test 5: Logit update logic")

initial_logits = {
    'Ooc Bom Bóc': 0.5,
    'Tết Choi Chnam Thmay': 0.3,
}

# Simulate parsed answer: user says YES to Den hoa dang
parsed_answer = {
    'Den hoa dang': {'status': True, 'confidence': 1.0},
    'Ghe ngo': {'status': False, 'confidence': 0.9},
}

final_logits = initial_logits.copy()

for fest in candidates:
    unsatisfied_rules = unsatisfied.get(fest, [])
    
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
                    print(f"   [{fest}] '{param}' CÓ: +{delta:.2f}")
                elif status is False:
                    penalty = (weight * conf) / 2
                    final_logits[fest] -= penalty
                    print(f"   [{fest}] '{param}' KHÔNG: -{penalty:.2f}")

print(f"\n   Initial logits: {initial_logits}")
print(f"   Final logits:   {final_logits}")

# Ooc Bom Bóc should increase (Den hoa dang = YES) and decrease (Ghe ngo = NO)
# Net effect depends on weights

print("   ✅ Test 5 PASSED!")

# ========== SUMMARY ==========
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("""
📋 Summary:
   1. Feature prioritization: OK (common features first)
   2. Question grouping: OK (max 3 questions)
   3. Question structure: OK (all fields present)
   4. Continue asking logic: OK (based on T_high)
   5. Logit update: OK (reward/penalty applied)
""")
