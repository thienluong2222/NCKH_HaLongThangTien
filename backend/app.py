from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from functools import wraps

from services import YOLOCSVPipeline, BayesianFestivalClassifier, sigmoid
from constraintsDB import CONSTRAINTS_DB, SUBCLASS_TO_FESTIVAL
from config import Config
from history_store import history_store
from models import (
    AnalysisHistory, AnalysisResult, DetectedObject, 
    QARecord, FestivalConstraints, ConstraintResult
)

# ==========================================
# SETUP LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# FLASK APP INITIALIZATION
# ==========================================
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Tạo thư mục upload nếu chưa có
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# KHỞI TẠO SERVICES
# ==========================================
logger.info("Đang khởi tạo services...")

# Kiểm tra configuration
config_errors = Config.validate()
if config_errors:
    for err in config_errors:
        logger.warning(f"⚠️ Config warning: {err}")

try:
    yolo_pipe = YOLOCSVPipeline(Config.MODEL_PATH, Config.CSV_PATH)
    classifier = BayesianFestivalClassifier(Config.GEMINI_API_KEY)
    logger.info("✅ Services khởi tạo thành công")
except Exception as e:
    logger.error(f"❌ Lỗi khởi tạo services: {e}")
    yolo_pipe = None
    classifier = None

# In-memory storage cho sessions đang xử lý
ACTIVE_SESSIONS = {}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def allowed_file(filename: str, file_type: str = 'video') -> bool:
    """Kiểm tra file extension có hợp lệ không"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'video':
        return ext in Config.ALLOWED_VIDEO_EXTENSIONS
    elif file_type == 'image':
        return ext in Config.ALLOWED_IMAGE_EXTENSIONS
    return False


def require_services(f):
    """Decorator kiểm tra services đã được khởi tạo"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if yolo_pipe is None or classifier is None:
            return jsonify({
                'error': 'Services chưa được khởi tạo. Kiểm tra model path và API key.',
                'code': 'SERVICE_NOT_INITIALIZED'
            }), 503
        return f(*args, **kwargs)
    return decorated_function


def detections_to_objects(detections) -> list:
    """Chuyển đổi detection objects sang list DetectedObject"""
    return [
        DetectedObject(
            subclass=d.subclass,
            confidence=round(d.confidence, 4),
            frame_id=d.frame_id,
            time_stamp=round(d.time_stamp, 2),
            count=d.count
        )
        for d in detections
    ]


# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    ---
    Kiểm tra trạng thái server và services
    """
    services_status = {
        'yolo_pipeline': yolo_pipe is not None,
        'bayesian_classifier': classifier is not None,
        'gemini_api_key': Config.GEMINI_API_KEY is not None
    }
    
    all_healthy = all(services_status.values())
    
    return jsonify({
        'status': 'healthy' if all_healthy else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'services': services_status,
        'history_count': history_store.count()
    }), 200 if all_healthy else 503


@app.route('/api/festivals', methods=['GET'])
def get_festivals():
    """
    Lấy danh sách tất cả lễ hội được hỗ trợ
    ---
    Returns:
        - festivals: Danh sách lễ hội với số lượng ràng buộc
    """
    festivals = []
    for festival_name, rules in CONSTRAINTS_DB.items():
        hard_constraints = sum(1 for r in rules if r[2])  # is_hard = True
        soft_constraints = sum(1 for r in rules if not r[2])
        
        festivals.append({
            'name': festival_name,
            'total_constraints': len(rules),
            'hard_constraints': hard_constraints,
            'soft_constraints': soft_constraints
        })
    
    return jsonify({
        'total': len(festivals),
        'festivals': festivals
    }), 200


@app.route('/api/video', methods=['POST'])
@require_services
def analyze_video():
    """
    Upload và phân tích video
    ---
    Request: multipart/form-data với field 'video'
    Returns:
        - status: 'needs_clarification' hoặc 'finished'
        - Nếu cần hỏi thêm: question, candidates_preliminary, request_id
        - Nếu hoàn thành: result, probabilities, top_3_constraints
    """
    if 'video' not in request.files:
        return jsonify({'error': 'Không có file video', 'code': 'NO_FILE'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Chưa chọn file', 'code': 'EMPTY_FILENAME'}), 400

    if not allowed_file(video_file.filename, 'video'):
        return jsonify({
            'error': 'Định dạng file không hỗ trợ',
            'code': 'INVALID_FORMAT',
            'allowed': list(Config.ALLOWED_VIDEO_EXTENSIONS)
        }), 400

    # Lưu file
    filename = secure_filename(video_file.filename)
    unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    video_file.save(video_path)
    logger.info(f"📹 Đã lưu video: {unique_filename}")

    try:
        # Step 1: YOLO Detection
        logger.info("🔍 Bắt đầu YOLO detection...")
        detected_objects = yolo_pipe.process_video(video_path)
        logger.info(f"Phát hiện {len(detected_objects)} objects")
        
        # Step 2: Tính toán Bayesian logits (ĐÃ CẬP NHẬT: trả về 3 giá trị)
        logger.info("Tính toán Bayesian logits...")
        logits, unsatisfied, satisfied = classifier.calculate_initial_logits(detected_objects)
        candidates = classifier.select_candidates(logits)
        
        # Lấy top 3 với constraints
        top_3_constraints = classifier.get_top_3_with_constraints(logits, satisfied, unsatisfied)
        
        # Tạo history record
        history = AnalysisHistory(
            filename=filename,
            detected_objects=detections_to_objects(detected_objects)
        )
        
        if not candidates:
            # Không xác định được lễ hội
            logger.info("Không xác định được lễ hội nào")
            
            # Lấy top 3 dù không có candidates
            probs = {f: sigmoid(l) for f, l in logits.items()}
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            top_3 = [{"festival": f, "confidence": round(p, 4)} for f, p in sorted_probs]
            
            history.result = AnalysisResult(
                winner=None,
                top_3=top_3,
                top_3_constraints=[]
            )
            history.status = "finished"
            history_store.save(history)
            
            return jsonify({
                "status": "finished",
                "history_id": history.id,
                "result": None,
                "message": "Không xác định được lễ hội nào từ video.",
                "top_3": top_3,
                "top_3_constraints": top_3_constraints,
                "detected_objects_count": len(detected_objects)
            }), 200
        
        # Sinh câu hỏi làm rõ
        qa_package = classifier.generate_consolidated_question(candidates, unsatisfied)
        
        if qa_package:
            # Cần hỏi thêm user
            logger.info(f"Cần hỏi thêm về {len(qa_package['target_features'])} features")
            
            req_id = history.id
            
            # Lưu session để xử lý tiếp
            ACTIVE_SESSIONS[req_id] = {
                "logits": logits,
                "unsatisfied": unsatisfied,
                "satisfied": satisfied,
                "candidates": candidates,
                "qa_package": qa_package,
                "video_path": video_path,
                "history": history
            }
            
            # Update history status
            history.status = "needs_clarification"
            history_store.save(history)
            
            return jsonify({
                "status": "needs_clarification",
                "request_id": req_id,
                "history_id": history.id,
                "question": qa_package['question_text'],
                "target_features": qa_package['target_features'],
                "candidates_preliminary": candidates,
                "top_3_constraints": top_3_constraints,
                "detected_objects_count": len(detected_objects)
            }), 200
        else:
            # Đã tự tin, không cần hỏi thêm
            logger.info("Đủ tự tin để kết luận")
            winners, final_probs = classifier.decide_final_result(logits)
            
            sorted_probs = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)[:3]
            top_3 = [{"festival": f, "confidence": round(p, 4)} for f, p in sorted_probs]
            
            history.result = AnalysisResult(
                winner=winners[0] if winners else None,
                top_3=top_3
            )
            history.status = "finished"
            history_store.save(history)
            
            return jsonify({
                "status": "finished",
                "history_id": history.id,
                "result": winners,
                "probabilities": {k: round(float(v), 4) for k, v in final_probs.items()},
                "top_3": top_3,
                "top_3_constraints": top_3_constraints,
                "detected_objects_count": len(detected_objects)
            }), 200
        
    except Exception as e:
        logger.error(f"Lỗi xử lý video: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Lỗi xử lý video: {str(e)}',
            'code': 'PROCESSING_ERROR'
        }), 500


@app.route('/api/answer', methods=['POST'])
@require_services
def submit_answer():
    """
    Gửi câu trả lời cho câu hỏi làm rõ
    ---
    Request JSON:
        - request_id: ID của phiên phân tích
        - answer: Câu trả lời của user
    """
    data = request.json
    if not data:
        return jsonify({'error': 'Thiếu request body', 'code': 'NO_BODY'}), 400
        
    req_id = data.get('request_id')
    user_answer = data.get('answer', '').strip()

    if not req_id or req_id not in ACTIVE_SESSIONS:
        return jsonify({
            'error': 'Request ID không hợp lệ hoặc đã hết hạn',
            'code': 'INVALID_REQUEST_ID'
        }), 400
    
    if not user_answer:
        return jsonify({'error': 'Câu trả lời không được để trống', 'code': 'EMPTY_ANSWER'}), 400

    session_data = ACTIVE_SESSIONS[req_id]
    logits = session_data['logits']
    unsatisfied = session_data['unsatisfied']
    satisfied = session_data['satisfied']
    candidates = session_data['candidates']
    qa_package = session_data['qa_package']
    history = session_data['history']

    try:
        logger.info(f"Phân tích câu trả lời: '{user_answer[:50]}...'")
        
        # Phân tích câu trả lời bằng LLM
        parsed_result = classifier.analyze_complex_answer(
            qa_package['question_text'],
            user_answer,
            qa_package['target_features']
        )
        
        # Cập nhật logits
        final_logits = classifier.update_logits_from_consolidated_answer(
            logits, candidates, unsatisfied, parsed_result
        )
        
        # Kết luận cuối cùng
        winners, final_probs = classifier.decide_final_result(final_logits)
        
        # Lấy top 3 với constraints cập nhật
        top_3_constraints = classifier.get_top_3_with_constraints(
            final_logits, satisfied, unsatisfied
        )
        
        sorted_probs = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3 = [{"festival": f, "confidence": round(p, 4)} for f, p in sorted_probs]
        
        # Cập nhật history
        history.qa_history.append(QARecord(
            question=qa_package['question_text'],
            answer=user_answer
        ))
        history.result = AnalysisResult(
            winner=winners[0] if winners else None,
            top_3=top_3
        )
        history.status = "finished"
        history_store.update(history.id, history)
        
        # Xóa session
        del ACTIVE_SESSIONS[req_id]
        logger.info(f"Hoàn thành phân tích: {winners}")
        
        return jsonify({
            "status": "finished",
            "history_id": history.id,
            "result": winners,
            "probabilities": {k: round(float(v), 4) for k, v in final_probs.items()},
            "top_3": top_3,
            "top_3_constraints": top_3_constraints,
            "analysis_breakdown": parsed_result
        }), 200

    except Exception as e:
        logger.error(f"Lỗi phân tích câu trả lời: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Lỗi phân tích: {str(e)}',
            'code': 'ANALYSIS_ERROR'
        }), 500


# ==========================================
# HISTORY ENDPOINTS
# ==========================================

@app.route('/api/history', methods=['GET'])
def get_history():
    """
    Lấy danh sách lịch sử phân tích
    ---
    Query params:
        - limit: Số lượng records (mặc định 50)
        - offset: Vị trí bắt đầu (mặc định 0)
    """
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    # Giới hạn limit
    limit = min(limit, 100)
    
    histories = history_store.get_all(limit=limit, offset=offset)
    
    return jsonify({
        'total': history_store.count(),
        'limit': limit,
        'offset': offset,
        'data': [h.to_summary() for h in histories]
    }), 200


@app.route('/api/history/<history_id>', methods=['GET'])
def get_history_detail(history_id):
    """
    Lấy chi tiết một phiên phân tích
    ---
    Params:
        - history_id: ID của phiên phân tích
    """
    history = history_store.get(history_id)
    
    if not history:
        return jsonify({
            'error': 'Không tìm thấy lịch sử phân tích',
            'code': 'NOT_FOUND'
        }), 404
    
    return jsonify(history.to_dict()), 200


@app.route('/api/history/<history_id>', methods=['DELETE'])
def delete_history(history_id):
    """
    Xóa một record lịch sử
    """
    success = history_store.delete(history_id)
    
    if not success:
        return jsonify({
            'error': 'Không tìm thấy lịch sử phân tích',
            'code': 'NOT_FOUND'
        }), 404
    
    return jsonify({
        'message': 'Đã xóa thành công',
        'deleted_id': history_id
    }), 200


@app.route('/api/history', methods=['DELETE'])
def delete_all_history():
    """
    Xóa toàn bộ lịch sử
    """
    count = history_store.delete_all()
    
    return jsonify({
        'message': f'Đã xóa {count} records',
        'deleted_count': count
    }), 200


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File quá lớn',
        'code': 'FILE_TOO_LARGE',
        'max_size_mb': Config.MAX_CONTENT_LENGTH // (1024 * 1024)
    }), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint không tồn tại',
        'code': 'NOT_FOUND'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal Server Error: {error}", exc_info=True)
    return jsonify({
        'error': 'Lỗi server nội bộ',
        'code': 'INTERNAL_ERROR'
    }), 500


# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    logger.info(f"Khởi động server tại http://{Config.HOST}:{Config.PORT}")
    app.run(
        debug=Config.DEBUG,
        host=Config.HOST,
        port=Config.PORT
    )