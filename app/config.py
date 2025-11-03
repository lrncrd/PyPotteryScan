"""
Configuration for PyPotteryScan Flask App
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

class Config:
    """Default Flask configuration"""
    
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    
    # Model directories
    MODELS_BASE_DIR = os.path.join(BASE_DIR, "models")
    OLMOCR_MODEL_DIR = os.path.join(MODELS_BASE_DIR, "olmocr-7b-fp4")
    QWEN_MODEL_DIR = os.path.join(MODELS_BASE_DIR, "qwen3-1.7b")
    
    # HuggingFace model IDs
    OLMOCR_MODEL_ID = "lrncrd/olmOCR-7B-FP4"
    QWEN_MODEL_ID = "Qwen/Qwen3-1.7B"
    
    # Server settings
    HOST = '0.0.0.0'
    PORT = 5002
    
    # File upload settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    
    # CORS settings
    CORS_ORIGINS = "*"
    
    @classmethod
    def get_config_dict(cls):
        """Return configuration as dictionary"""
        return {
            'MODELS_BASE_DIR': cls.MODELS_BASE_DIR,
            'OLMOCR_MODEL_DIR': cls.OLMOCR_MODEL_DIR,
            'QWEN_MODEL_DIR': cls.QWEN_MODEL_DIR,
            'OLMOCR_MODEL_ID': cls.OLMOCR_MODEL_ID,
            'QWEN_MODEL_ID': cls.QWEN_MODEL_ID
        }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # In production, set proper CORS origins
    CORS_ORIGINS = ["http://localhost:5002"]
