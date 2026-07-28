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
    
    # Model directories - shared across the suite when launched by the PyPottery
    # Suite launcher (PYPOTTERY_MODEL_CACHE set, so a model already downloaded by
    # another app isn't fetched again, and updating this app doesn't wipe the
    # already-downloaded models). Standalone (no launcher), self-contained as before.
    _suite_model_cache = os.environ.get('PYPOTTERY_MODEL_CACHE')
    MODELS_BASE_DIR = os.path.join(_suite_model_cache, "scan") if _suite_model_cache else os.path.join(BASE_DIR, "models")
    QWEN_MODEL_DIR = os.path.join(MODELS_BASE_DIR, "qwen3-1.7b")
    
    # OlmOCR Model Options
    # FP4 (4-bit): ~5GB, requires NVIDIA GPU + CUDA
    OLMOCR_FP4_MODEL_ID = "lrncrd/olmOCR-7B-FP4"
    OLMOCR_FP4_MODEL_DIR = os.path.join(MODELS_BASE_DIR, "olmocr-7b-fp4")

    # Selected model persistence file
    SELECTED_MODEL_FILE = os.path.join(MODELS_BASE_DIR, "selected_model.txt")
    
    # Projects directory
    PROJECTS_DIR = os.path.join(BASE_DIR, "projects")
    
    # HuggingFace model IDs (Qwen for parsing)
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
            'QWEN_MODEL_DIR': cls.QWEN_MODEL_DIR,
            'QWEN_MODEL_ID': cls.QWEN_MODEL_ID,
            'PROJECTS_DIR': cls.PROJECTS_DIR,
            # OlmOCR FP4 (4-bit)
            'OLMOCR_FP4_MODEL_ID': cls.OLMOCR_FP4_MODEL_ID,
            'OLMOCR_FP4_MODEL_DIR': cls.OLMOCR_FP4_MODEL_DIR,
            # Selection persistence
            'SELECTED_MODEL_FILE': cls.SELECTED_MODEL_FILE,
        }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # In production, set proper CORS origins
    CORS_ORIGINS = ["http://localhost:5002"]
