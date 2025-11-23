import streamlit as st
import numpy as np
import json
import os
from collections import defaultdict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# PHẦN 1: CẤU TRÚC DỮ LIỆU & RÀNG BUỘC (CSP)
# ==========================================

class ObjectDetection:
    """Cấu trúc dữ liệu đại diện cho một đối tượng được phát hiện (hoặc người dùng thêm vào)"""
    def __init__(self, subclass, confidence, frame_id, count, is_virtual=False):
        self.subclass = subclass
        self.confidence = confidence
        self.frame_id = frame_id
        self.count = count
        self.is_virtual = is_virtual  # True nếu do người dùng xác nhận, False nếu do YOLO detect

# Database Ràng buộc (Trích xuất từ Colab của bạn)
CONSTRAINTS_DB = {
    "Lễ hội Ooc Bom Bóc": [
        ("confidence_min", ["Ghe ngo"], True, 1.0, 0.7),
        ("is_presence", ["Den hoa dang", "Den nuoc"], True, 1.0, None),
        ("at_least", ["Den troi"], False, 0.5, 1),
        ("at_least", ["Com", "Khoai"], False, 0.6, None)
    ],
    "Tết Choi Chnam Thmay": [
        ("confidence_min", ["Nui cat"], True, 1.0, 0.8),
        ("at_least", ["Nguoi tham gia te nuoc", "Nuoc thom"], False, 0.8, None),
        ("is_presence", ["Tuong Phat", "Nha su"], True, 1.0, None)
    ],
    "Chợ nổi Cái Răng": [
        ("is_presence", ["Cay beo", "Thuyen"], True, 1.0, None),
        ("is_on", ["Cay beo", "Thuyen"], True, 1.0, None)
    ]
}

# Danh sách tất cả các subclass hệ thống biết
ALL_SUBCLASSES = [
    "Ghe ngo", "Den hoa dang", "Den nuoc", "Den troi", "Com", "Khoai",
    "Nui cat", "Nguoi tham gia te nuoc", "Nuoc thom", "Tuong Phat", "Nha su",
    "Cay beo", "Thuyen"
]

def check_constraints(detections, constraints_db, score_threshold=0.50):
    """
    Hàm kiểm tra xem danh sách detections hiện tại khớp với lễ hội nào nhất.
    """
    detections_by_subclass = defaultdict(list)
    for det in detections:
        detections_by_subclass[det.subclass].append(det)

    festival_results = {}

    for festival, constraints in constraints_db.items():
        total_score = 0.0
        max_score = 0.0
        hard_failed = False
        missing_rules = []

        for constraint in constraints:
            ctype, params, is_hard, weight, threshold = constraint
            satisfied = False

            if ctype == "is_presence" or ctype == "is_presence_in_frame":
                satisfied = all(sub in detections_by_subclass for sub in params)
            elif ctype == "at_least" or ctype == "at_least_in_frame":
                satisfied = all(sub in detections_by_subclass for sub in params)
            elif ctype == "confidence_min":
                target = params[0]
                if target in detections_by_subclass:
                    confs = [d.confidence for d in detections_by_subclass[target]]
                    avg_conf = np.mean(confs)
                    satisfied = avg_conf >= threshold
                else:
                    satisfied = False
            elif ctype == "is_on":
                satisfied = all(sub in detections_by_subclass for sub in params)

            if satisfied:
                total_score += weight
            else:
                missing_rules.append((ctype, params))
                if is_hard:
                    hard_failed = True
            
            max_score += weight

        normalized_score = total_score / max_score if max_score > 0 else 0.0
        final_valid_score = 0.0 if hard_failed else normalized_score

        festival_results[festival] = {
            "score": final_valid_score,
            "potential_score": normalized_score,
            "missing": missing_rules,
            "hard_failed": hard_failed
        }

    best_candidate = max(festival_results, key=lambda k: festival_results[k]['potential_score'])
    
    return {
        "best_candidate": best_candidate,
        "current_score": festival_results[best_candidate]['score'],
        "potential_score": festival_results[best_candidate]['potential_score'],
        "details": festival_results
    }

# ==========================================
# PHẦN 2: TRÍ TUỆ NHÂN TẠO (LLM FUNCTIONS - ĐÃ SỬA LỖI)
# ==========================================

def load_knowledge():
    if os.path.exists("text_data.txt"):
        loader = TextLoader("text_data.txt", encoding='utf-8')
        return loader.load()[0].page_content
    return ""

def generate_verification_question(candidate, missing_rule, knowledge_text, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    
    ctype, params = missing_rule
    missing_objects = ", ".join(params)
    
    prompt_template = """
    Bạn là một thám tử AI. Bạn đang nghi ngờ video này là về lễ hội: "{candidate}".
    Tuy nhiên, hệ thống máy tính chưa tìm thấy hình ảnh của: {missing_objects}.
    
    Sử dụng kiến thức sau đây:
    {context}
    
    Hãy đặt một câu hỏi ngắn gọn, lịch sự cho người dùng để xác nhận xem họ có nhìn thấy vật thể đó trong video không.
    Ví dụ: "Bạn có thấy chiếc ghe ngo (thuyền dài) nào xuất hiện không?"
    Chỉ in ra câu hỏi.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Tạo chain: Prompt -> LLM -> String Output
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "candidate": candidate,
        "missing_objects": missing_objects,
        "context": knowledge_text
    })
    return response

def analyze_user_answer(user_text, expected_objects, api_key):

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
    
    prompt_template = """
    Hệ thống đang hỏi người dùng về việc họ có thấy các vật thể sau không: {expected_objects}.
    Người dùng trả lời: "{user_text}".
    Danh sách các ID vật thể hợp lệ trong hệ thống: {all_subclasses}.
    
    Hãy phân tích câu trả lời:
    1. Người dùng có xác nhận (YES) là nhìn thấy không?
    2. Nếu có, họ đang nói về vật thể nào trong danh sách ID hợp lệ?
    
    Trả về kết quả dưới dạng JSON KHÔNG CÓ Markdown:
    {{
        "is_confirmed": true/false,
        "detected_object_id": "Ten_Subclass_Hoac_Null"
    }}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Tạo chain
    chain = prompt | llm | StrOutputParser()
    
    response_str = chain.invoke({
        "expected_objects": ", ".join(expected_objects),
        "user_text": user_text,
        "all_subclasses": ", ".join(ALL_SUBCLASSES)
    })
    
    # Xử lý JSON
    try:
        content = response_str.strip().replace("```json", "").replace("```", "")
        return json.loads(content)
    except:
        return {"is_confirmed": False, "detected_object_id": None}

# ==========================================
# PHẦN 3: GIAO DIỆN STREAMLIT
# ==========================================

st.set_page_config(page_title="AI Human-in-the-loop Chatbot", page_icon="🕵️")

st.title("🕵️ Hệ thống xác thực Lễ hội thông minh")
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.header("Cấu hình")
    api_key = st.text_input("Nhập Gemini API Key", type="default", value="AIzaSyAyAxK7QfgwcETsoLZ3iB4SbUYP0gTGSCg")
    
    if st.button("🔄 Reset dữ liệu giả lập"):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Trạng thái hiện tại:**")
    if "detections" in st.session_state:
        st.write(f"Số lượng vật thể: {len(st.session_state.detections)}")
        for det in st.session_state.detections:
            icon = "👤" if det.is_virtual else "📷"
            st.code(f"{icon} {det.subclass} ({det.confidence:.1f})")

# --- Khởi tạo dữ liệu ---
if "detections" not in st.session_state:
    st.session_state.detections = [
        ObjectDetection("Den nuoc", 0.95, 1, 1),
        ObjectDetection("Com", 0.75, 5, 1)
    ]
    st.session_state.chat_history = []
    st.session_state.finished = False
    
    initial_msg = "Chào bạn! Tôi đã phân tích video. Tôi thấy có **Đèn nước** và **Cốm**. Tuy nhiên, tôi chưa chắc chắn đây là lễ hội gì."
    st.session_state.chat_history.append({"role": "assistant", "content": initial_msg})

# --- Hiển thị lịch sử chat ---
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Main Logic Loop ---
if not st.session_state.finished and api_key:
    
    result = check_constraints(st.session_state.detections, CONSTRAINTS_DB)
    best_cand = result["best_candidate"]
    curr_score = result["current_score"]
    missing = result["details"][best_cand]["missing"]
    
    if curr_score >= 0.85:
        success_msg = f"🎉 **KẾT LUẬN:** Dựa trên các bằng chứng (cả từ camera và xác nhận của bạn), tôi khẳng định đây là **{best_cand}** (Độ tin cậy: {curr_score:.0%})."
        if st.session_state.chat_history[-1]["content"] != success_msg:
            st.chat_message("assistant").write(success_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": success_msg})
            st.session_state.finished = True
            st.balloons()
            st.rerun()
    
    else:
        last_role = st.session_state.chat_history[-1]["role"]
        
        if last_role == "user" or len(st.session_state.chat_history) == 1:
            if not missing:
                warn_msg = "Không còn manh mối nào để hỏi, nhưng độ tin cậy vẫn thấp."
                if st.session_state.chat_history[-1]["content"] != warn_msg:
                    st.warning(warn_msg)
                    st.session_state.finished = True
            else:
                missing_rule = missing[0]
                missing_params = missing_rule[1]
                
                with st.spinner("Đang suy luận câu hỏi tiếp theo..."):
                    knowledge = load_knowledge()
                    question = generate_verification_question(best_cand, missing_rule, knowledge, api_key)
                
                st.chat_message("assistant").write(question)
                st.session_state.chat_history.append({"role": "assistant", "content": question})
                st.session_state.pending_check = missing_params

# --- Input Box ---
if not st.session_state.finished:
    if prompt := st.chat_input("Nhập câu trả lời của bạn..."):
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        if "pending_check" in st.session_state and api_key:
            with st.spinner("Đang phân tích câu trả lời..."):
                analysis = analyze_user_answer(prompt, st.session_state.pending_check, api_key)
                
                if analysis["is_confirmed"] and analysis["detected_object_id"]:
                    obj_id = analysis["detected_object_id"]
                    new_obj = ObjectDetection(subclass=obj_id, confidence=1.0, frame_id=-1, count=1, is_virtual=True)
                    st.session_state.detections.append(new_obj)
                    st.caption(f"✅ *Đã ghi nhận bằng chứng mới:* **{obj_id}**")
                else:
                    st.caption("Đã ghi nhận phản hồi (Không tìm thấy bằng chứng mới).")
                
                st.session_state.pop("pending_check", None)
                st.rerun()
        else:
            if not api_key:
                st.error("Vui lòng nhập API Key để tiếp tục.")