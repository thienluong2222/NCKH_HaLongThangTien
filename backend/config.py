import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Cấu hình chung cho ứng dụng"""
    
    # Flask
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    PORT = int(os.getenv("PORT", 5001))
    HOST = os.getenv("HOST", "0.0.0.0")
    
    # Upload
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
    
    # Model paths
    MODEL_PATH = os.getenv("MODEL_PATH", "weight/best.pt")
    CSV_PATH = os.getenv("CSV_PATH", "uploads/artifacts/merged_data.csv")
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Bayesian Config
    BAYESIAN_CONFIG = {
        "T_high": 0.85,    # Ngưỡng tin cậy cao
        "T_low": 0.50,     # Ngưỡng thấp nhất
        "delta": 0.25,     # Chênh lệch tối đa
        "T_out": 0.85      # Ngưỡng quyết định cuối
    }
    
    @classmethod
    def validate(cls):
        """Kiểm tra các cấu hình bắt buộc"""
        errors = []
        
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY chưa được cấu hình trong .env")
        
        if not os.path.exists(cls.MODEL_PATH):
            errors.append(f"Model không tồn tại: {cls.MODEL_PATH}")
            
        if not os.path.exists(cls.CSV_PATH):
            errors.append(f"CSV mapping không tồn tại: {cls.CSV_PATH}")
            
        return errors
