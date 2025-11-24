import streamlit as st
import numpy as np
import os
import tempfile
import json
from collections import defaultdict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Import các module từ source hiện có
from model import ObjectDetection, YOLOCSVPipeline, calculate_initial_score, get_unsatisfied_constraints, generate_question, analyze_user_response
from constraintsDB import CONSTRAINTS_DB, SUBCLASS_TO_FESTIVAL

# --- CẤU HÌNH ---
load_dotenv()
MODEL_PATH = "../models/best.pt" 
CSV_PATH = "../artifacts/merged_data.csv"
FINAL_SCORE_THRESHOLD = 0.85  # Ngưỡng điểm để chốt kết quả


# ==========================================
#  UI & APP FLOW
# ==========================================

st.set_page_config(page_title="Chatbot Lễ hội Logic", page_icon="🤖")
st.title("🤖 Xác thực Lễ hội (Human-in-the-Loop)")

# --- Session State Init ---
if "detections" not in st.session_state: st.session_state.detections = []
if "candidates_queue" not in st.session_state: st.session_state.candidates_queue = [] # Danh sách ứng viên cần xét
if "current_candidate_idx" not in st.session_state: st.session_state.current_candidate_idx = 0
if "user_confirmed_weight" not in st.session_state: st.session_state.user_confirmed_weight = 0.0 # Điểm cộng thêm từ user
if "rejected_rules" not in st.session_state: st.session_state.rejected_rules = set() # Các luật user bảo KHÔNG
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "app_state" not in st.session_state: st.session_state.app_state = "UPLOAD" # UPLOAD -> PROCESSING -> VERIFYING -> FINISHED
if "current_unsatisfied_rules" not in st.session_state: st.session_state.current_unsatisfied_rules = []

# --- Sidebar ---
with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
    
    if uploaded_file and st.button("Bắt đầu Phân tích"):
        st.session_state.chat_history = []
        st.session_state.user_confirmed_weight = 0.0
        st.session_state.rejected_rules = set()
        st.session_state.current_candidate_idx = 0
        st.session_state.app_state = "PROCESSING"
        
        # 1. Lưu và xử lý video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        try:
            pipeline = YOLOCSVPipeline(MODEL_PATH, CSV_PATH)
            with st.spinner("Đang chạy YOLO detection..."):
                # Chạy pipeline để lấy detections
                summary = pipeline.process_video_with_output(tfile.name, output_path='../assets/output/temp_output.mp4', save_frames=False)
                
                # Convert kết quả về ObjectDetection objects
                raw_detections = []
                if summary and 'frame_details' in summary:
                    for frame in summary['frame_details']:
                        for det in frame['detections']:
                            if det.get('mapped', False):
                                obj = ObjectDetection(
                                    subclass=det['label'],
                                    confidence=det['confidence'],
                                    frame_id=frame['frame'],
                                    time_stamp=frame['time'],
                                    count=1, bboxs=[det['box']]
                                )
                                raw_detections.append(obj)
                st.session_state.detections = raw_detections
                
                # 2. Lập danh sách ứng viên (Candidate Queue)
                # Tính điểm sơ bộ cho TẤT CẢ lễ hội để sắp xếp thứ tự ưu tiên
                candidates = []
                all_festivals = CONSTRAINTS_DB.keys()
                
                # Lọc nhanh bằng SUBCLASS_TO_FESTIVAL nếu muốn, hoặc duyệt hết
                # Ở đây duyệt hết cho chắc chắn
                for fest in all_festivals:
                    achieved, possible = calculate_initial_score(fest, raw_detections)
                    normalized = achieved / possible if possible > 0 else 0
                    candidates.append({
                        "name": fest,
                        "initial_score": achieved,
                        "total_possible": possible,
                        "normalized": normalized
                    })
                
                # Sắp xếp giảm dần theo điểm normalized
                candidates.sort(key=lambda x: x['normalized'], reverse=True)
                st.session_state.candidates_queue = candidates
                
                # Chuyển sang trạng thái Verify
                st.session_state.app_state = "VERIFYING"
                st.session_state.current_unsatisfied_rules = [] # Reset cho ứng viên đầu tiên
                st.rerun()

        except Exception as e:
            st.error(f"Lỗi xử lý: {str(e)}")

# --- Main Logic Area ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Hội thoại xác thực")
    
    # Hiển thị lịch sử chat
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    # LOGIC CHÍNH: QUẢN LÝ QUY TRÌNH HỎI ĐÁP
    if st.session_state.app_state == "VERIFYING":
        
        # 1. Kiểm tra xem còn ứng viên nào không
        if st.session_state.current_candidate_idx >= len(st.session_state.candidates_queue):
            st.warning("⚠️ Đã kiểm tra hết các lễ hội tiềm năng nhưng không có kết quả nào đạt ngưỡng.")
            st.session_state.app_state = "FINISHED"
            st.stop()

        # 2. Lấy thông tin ứng viên hiện tại
        candidate_data = st.session_state.candidates_queue[st.session_state.current_candidate_idx]
        candidate_name = candidate_data['name']
        total_possible = candidate_data['total_possible']
        
        # 3. Tính điểm hiện tại (Base + User Confirmed)
        current_achieved = candidate_data['initial_score'] + st.session_state.user_confirmed_weight
        current_normalized = current_achieved / total_possible if total_possible > 0 else 0
        
        # 4. Kiểm tra ngưỡng
        if current_normalized >= FINAL_SCORE_THRESHOLD:
            st.balloons()
            success_msg = f"🎉 **KẾT LUẬN:** Video này quay **{candidate_name}**!\n\nĐộ tin cậy: {current_normalized:.1%} (Đã đạt ngưỡng > {FINAL_SCORE_THRESHOLD:.1%})"
            st.session_state.chat_history.append({"role": "assistant", "content": success_msg})
            st.session_state.app_state = "FINISHED"
            st.rerun()
            
        # 5. Nếu chưa đạt ngưỡng, chuẩn bị câu hỏi
        else:
            # Lấy danh sách luật chưa thỏa mãn (chỉ lấy 1 lần đầu mỗi khi chuyển candidate)
            if not st.session_state.current_unsatisfied_rules:
                missing = get_unsatisfied_constraints(candidate_name, st.session_state.detections)
                # Lọc bỏ các rule đã bị user từ chối (nếu có logic global, hiện tại xét local)
                st.session_state.current_unsatisfied_rules = missing
            
            # Lọc lại: Loại bỏ rule đã nằm trong rejected_rules của phiên này
            # (Lưu ý: rejected_rules cần reset khi đổi candidate nếu rule đó đặc thù, 
            # nhưng ở đây ta giả sử rule unique object thì nếu ko thấy là ko thấy luôn)
            # -> Cách tiếp cận đơn giản: rejected_rules là danh sách index hoặc object hash trong phiên hỏi này
            
            valid_rules_to_ask = []
            for r in st.session_state.current_unsatisfied_rules:
                # Rule structure: (type, params, is_hard, weight, threshold)
                # Dùng str(r) làm ID tạm thời để check đã reject chưa
                if str(r) not in st.session_state.rejected_rules:
                    valid_rules_to_ask.append(r)
            
            if not valid_rules_to_ask:
                # Hết câu hỏi mà vẫn chưa đủ điểm -> THẤT BẠI với ứng viên này
                fail_msg = f"❌ Không phải **{candidate_name}**. (Điểm: {current_normalized:.1%}). Đang xét khả năng tiếp theo..."
                st.session_state.chat_history.append({"role": "assistant", "content": fail_msg})
                
                # Chuyển sang ứng viên kế tiếp
                st.session_state.current_candidate_idx += 1
                st.session_state.user_confirmed_weight = 0.0 # Reset điểm bonus
                st.session_state.rejected_rules = set()      # Reset list từ chối
                st.session_state.current_unsatisfied_rules = [] # Clear cache rule
                st.rerun()
            else:
                # Vẫn còn câu hỏi -> Hỏi câu đầu tiên trong list
                rule_to_ask = valid_rules_to_ask[0]
                
                # Kiểm tra xem câu hỏi này đã được hiển thị chưa (tránh gen lại khi rerun)
                last_msg = st.session_state.chat_history[-1] if st.session_state.chat_history else None
                is_waiting_user = (last_msg and last_msg["role"] == "assistant" and "?" in last_msg["content"])
                
                if not is_waiting_user:
                    if not api_key:
                        st.error("Cần Gemini API Key để sinh câu hỏi.")
                        st.stop()
                        
                    with st.spinner(f"Đang phân tích {candidate_name}..."):
                        question = generate_question(candidate_name, rule_to_ask, api_key)
                        st.session_state.chat_history.append({"role": "assistant", "content": question})
                        st.rerun()

    # 6. Xử lý Input của User
    if prompt := st.chat_input("Trả lời (Có/Không)..."):
        if st.session_state.app_state == "VERIFYING":
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Lấy rule đang hỏi (Rule đầu tiên trong valid list)
            # Phải tính lại valid list để đảm bảo đồng bộ
            missing = st.session_state.current_unsatisfied_rules
            valid_rules = [r for r in missing if str(r) not in st.session_state.rejected_rules]
            
            if valid_rules:
                current_rule = valid_rules[0]
                rule_weight = current_rule[3]
                last_question = st.session_state.chat_history[-2]["content"] # Lấy câu hỏi của AI
                
                # Phân tích câu trả lời
                intent = analyze_user_response(last_question, prompt, api_key)
                
                if intent == "YES":
                    # CỘNG ĐIỂM TRỰC TIẾP
                    st.session_state.user_confirmed_weight += rule_weight
                    # Đánh dấu rule này coi như đã xong (bằng cách xóa khỏi unsatisfied hoặc logic khác)
                    # Ở đây ta dùng mẹo: Thêm vào rejected_rules? Không, rejected là bỏ đi.
                    # Ta cần loại nó khỏi danh sách cần hỏi.
                    # Cách đơn giản: Xóa khỏi st.session_state.current_unsatisfied_rules
                    st.session_state.current_unsatisfied_rules.remove(current_rule)
                    st.toast(f"✅ Đã xác nhận! (+{rule_weight} điểm)")
                    
                elif intent == "NO":
                    # XÓA RÀNG BUỘC (Thực chất là thêm vào blacklist để lần sau ko hỏi nữa)
                    st.session_state.rejected_rules.add(str(current_rule))
                    st.toast("❌ Đã loại bỏ ràng buộc này.")
                    
                else:
                    st.toast("🤔 Chưa rõ ý bạn, vui lòng trả lời Có hoặc Không.")
                
                st.rerun()

with col2:
    if st.session_state.app_state in ["VERIFYING", "FINISHED"] and st.session_state.candidates_queue:
        idx = st.session_state.current_candidate_idx
        if idx < len(st.session_state.candidates_queue):
            curr = st.session_state.candidates_queue[idx]
            
            st.info(f"🧐 Đang xét: **{curr['name']}**")
            
            # Tính điểm real-time
            total = curr['total_possible']
            base = curr['initial_score']
            bonus = st.session_state.user_confirmed_weight
            current_score = (base + bonus) / total if total > 0 else 0
            
            st.metric("Điểm hiện tại", f"{current_score:.1%}", f"Mục tiêu: {FINAL_SCORE_THRESHOLD:.1%}")
            
            st.write("---")
            st.write("📊 **Chi tiết điểm:**")
            st.write(f"- Điểm từ Video (AI): {base:.2f}")
            st.write(f"- Điểm User xác nhận: +{bonus:.2f}")
            st.write(f"- Tổng trọng số cần thiết: {total:.2f}")
            
            st.write("---")
            st.write("📋 **Hàng đợi ứng viên:**")
            for i, cand in enumerate(st.session_state.candidates_queue):
                icon = "🟢" if i == idx else "⚪" if i > idx else "🔴"
                st.text(f"{icon} {cand['name']} ({cand['normalized']:.1%})")