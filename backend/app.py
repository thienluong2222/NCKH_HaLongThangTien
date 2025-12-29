from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from services import YOLOCSVPipeline, BayesianFestivalClassifier # Import class của bạn
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'weight/best.pt'      
CSV_PATH = 'uploads/artifacts/merged_data.csv'
API_KEY = os.getenv("GEMINI_API_KEY")
yolo_pipe = YOLOCSVPipeline(MODEL_PATH, CSV_PATH)
classifier = BayesianFestivalClassifier(API_KEY)

MEMORY_STORAGE = {}

@app.route('/api/video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(video_path)

    try:
        detected_objects = yolo_pipe.process_video(video_path)
        logits, unsatisfied = classifier.calculate_initial_logits(detected_objects)
        candidates = classifier.select_candidates(logits)

        if not candidates:
            return jsonify({
                "status": "finished",
                "result": "Không xác định được lễ hội nào.",
                "details": {}
            }), 200
        
        qa_package = classifier.generate_consolidated_question(candidates, unsatisfied)

        if qa_package:
            req_id = str(uuid.uuid4())
            
            MEMORY_STORAGE[req_id] = {
                "logits": logits,
                "unsatisfied": unsatisfied,
                "candidates": candidates,
                "qa_package": qa_package,
                "video_path": video_path
            }
            return jsonify({
                "status": "needs_clarification",
                "request_id": req_id,
                "question": qa_package['question_text'],
                "candidates_preliminary": candidates
            }), 200
        else:
            winners, final_probs = classifier.decide_final_result(logits)
            return jsonify({
                "status": "finished",
                "result": winners,
                "probabilities": {k: float(v) for k, v in final_probs.items()}
            }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/answer', methods=['POST'])
def submit_answer():
    data = request.json
    req_id = data.get('request_id')
    user_answer = data.get('answer')

    if not req_id or req_id not in MEMORY_STORAGE:
        return jsonify({'error': 'Invalid request ID'}), 400

    session_data = MEMORY_STORAGE[req_id]
    logits = session_data['logits']
    unsatisfied = session_data['unsatisfied']
    candidates = session_data['candidates']
    qa_package = session_data['qa_package']

    try:
        parsed_result = classifier.analyze_complex_answer(
            qa_package['question_text'],
            user_answer,
            qa_package['target_features']
        )
        
        final_logits = classifier.update_logits_from_consolidated_answer(logits, candidates, unsatisfied, parsed_result)

        winners, final_probs = classifier.decide_final_result(final_logits)
        del MEMORY_STORAGE[req_id]
        return jsonify({
            "status": "finished",
            "result": winners,
            "probabilities": {k: float(v) for k, v in final_probs.items()},
            "analysis_breakdown": parsed_result
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)