# ==========================================
# ICHeritage Backend - API cURL Examples
# ==========================================
# Base URL: http://localhost:5001

# ==========================================
# 1. HEALTH CHECK
# ==========================================
# Kiểm tra trạng thái server và services

curl -X GET http://localhost:5001/api/health

# Response example:
# {
#   "status": "healthy",
#   "timestamp": "2026-01-02T10:00:00",
#   "services": {
#     "yolo_pipeline": true,
#     "bayesian_classifier": true,
#     "gemini_api_key": true
#   },
#   "history_count": 5
# }


# ==========================================
# 2. GET FESTIVALS LIST
# ==========================================
# Lấy danh sách tất cả lễ hội được hỗ trợ

curl -X GET http://localhost:5001/api/festivals

# Response example:
# {
#   "total": 5,
#   "festivals": [
#     {
#       "name": "Ok Om Bok",
#       "total_constraints": 12,
#       "hard_constraints": 5,
#       "soft_constraints": 7
#     },
#     ...
#   ]
# }


# ==========================================
# 3. ANALYZE VIDEO
# ==========================================
# Upload và phân tích video để nhận diện lễ hội
# Supported formats: mp4, avi, mov, mkv, webm

curl -X POST http://localhost:5001/api/video \
  -F "video=@/path/to/your/video.mp4"

# Windows PowerShell:
# curl.exe -X POST http://localhost:5001/api/video -F "video=@C:\path\to\video.mp4"

# Response khi cần hỏi thêm:
# {
#   "status": "needs_clarification",
#   "request_id": "abc123",
#   "history_id": "hist_xyz",
#   "question": "Trong video có đèn hoa đăng không? Có hoạt động đua ghe ngo không?",
#   "target_features": ["đèn hoa đăng", "ghe ngo"],
#   "candidates_preliminary": ["Ok Om Bok", "Chol Chnam Thmay"],
#   "top_3_constraints": [...],
#   "detected_objects_count": 15
# }

# Response khi đủ tự tin:
# {
#   "status": "finished",
#   "history_id": "hist_xyz",
#   "result": ["Ok Om Bok"],
#   "probabilities": {"Ok Om Bok": 0.85, "Chol Chnam Thmay": 0.12, ...},
#   "top_3": [
#     {"festival": "Ok Om Bok", "confidence": 0.85},
#     {"festival": "Chol Chnam Thmay", "confidence": 0.12},
#     {"festival": "Tết Nguyên Đán", "confidence": 0.03}
#   ],
#   "top_3_constraints": [...],
#   "detected_objects_count": 15
# }


# ==========================================
# 4. SUBMIT ANSWER (Human-in-the-Loop)
# ==========================================
# Gửi câu trả lời cho câu hỏi làm rõ
# Sử dụng request_id từ response của /api/video

curl -X POST http://localhost:5001/api/answer \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "abc123",
    "answer": "Có, tôi thấy đèn hoa đăng trôi trên sông và có hoạt động đua ghe ngo"
  }'

# Response:
# {
#   "status": "finished",
#   "history_id": "hist_xyz",
#   "result": ["Ok Om Bok"],
#   "probabilities": {"Ok Om Bok": 0.92, ...},
#   "top_3": [...],
#   "top_3_constraints": [...],
#   "analysis_breakdown": {
#     "đèn hoa đăng": true,
#     "ghe ngo": true
#   }
# }


# ==========================================
# 5. GET HISTORY LIST
# ==========================================
# Lấy danh sách lịch sử phân tích

# Lấy 50 records đầu tiên (mặc định)
curl -X GET http://localhost:5001/api/history

# Phân trang
curl -X GET "http://localhost:5001/api/history?limit=10&offset=0"

# Response:
# {
#   "total": 25,
#   "limit": 10,
#   "offset": 0,
#   "data": [
#     {
#       "id": "hist_xyz",
#       "filename": "video1.mp4",
#       "status": "finished",
#       "winner": "Ok Om Bok",
#       "created_at": "2026-01-02T10:00:00"
#     },
#     ...
#   ]
# }


# ==========================================
# 6. GET HISTORY DETAIL
# ==========================================
# Lấy chi tiết một phiên phân tích

curl -X GET http://localhost:5001/api/history/hist_xyz

# Response:
# {
#   "id": "hist_xyz",
#   "filename": "video1.mp4",
#   "status": "finished",
#   "detected_objects": [...],
#   "qa_history": [
#     {
#       "question": "Trong video có đèn hoa đăng không?",
#       "answer": "Có, tôi thấy đèn hoa đăng",
#       "timestamp": "2026-01-02T10:01:00"
#     }
#   ],
#   "result": {
#     "winner": "Ok Om Bok",
#     "top_3": [...]
#   },
#   "created_at": "2026-01-02T10:00:00",
#   "updated_at": "2026-01-02T10:01:30"
# }


# ==========================================
# 7. DELETE SINGLE HISTORY
# ==========================================
# Xóa một record lịch sử

curl -X DELETE http://localhost:5001/api/history/hist_xyz

# Response:
# {
#   "message": "Đã xóa thành công",
#   "deleted_id": "hist_xyz"
# }


# ==========================================
# 8. DELETE ALL HISTORY
# ==========================================
# Xóa toàn bộ lịch sử (cẩn thận!)

curl -X DELETE http://localhost:5001/api/history

# Response:
# {
#   "message": "Đã xóa 25 records",
#   "deleted_count": 25
# }


# ==========================================
# ERROR RESPONSES
# ==========================================
# 
# 400 Bad Request:
# {"error": "...", "code": "NO_FILE|EMPTY_FILENAME|INVALID_FORMAT|..."}
#
# 404 Not Found:
# {"error": "Không tìm thấy...", "code": "NOT_FOUND"}
#
# 413 Payload Too Large:
# {"error": "File quá lớn", "code": "FILE_TOO_LARGE", "max_size_mb": 500}
#
# 500 Internal Server Error:
# {"error": "Lỗi server nội bộ", "code": "INTERNAL_ERROR"}
#
# 503 Service Unavailable:
# {"error": "Services chưa được khởi tạo...", "code": "SERVICE_NOT_INITIALIZED"}
