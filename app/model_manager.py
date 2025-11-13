"""
Model loading and management for PyPotteryScan
"""
import os
import time
import logging
import threading
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages OlmOCR and Qwen models with intelligent memory management"""
    
    def __init__(self, config=None):
        self.config = config  # Store config reference
        self.processor = None
        self.model = None
        self.qwen_tokenizer = None
        self.qwen_model = None
        
        # Timestamps for auto-unload
        self.olmocr_last_used = None
        self.qwen_last_used = None
        self.unload_timeout = 10  # Seconds of inactivity before auto-unload
        
        # Timer threads
        self.olmocr_timer = None
        self.qwen_timer = None
        
        self.loading_status = {
            'stage': 'starting',
            'message': 'Initializing server...',
            'progress': 0
        }
        self.parsing_status = {
            'active': False,
            'current': 0,
            'total': 0,
            'current_line': ''
        }
    
    def get_loading_status(self):
        """Get current loading status"""
        return self.loading_status
    
    def get_parsing_status(self):
        """Get current parsing status"""
        return self.parsing_status
    
    def download_model_with_progress(self, model_id, local_dir, model_name, start_progress=10, end_progress=50):
        """Download model from HuggingFace with progress tracking"""
        try:
            logger.info(f"📥 Downloading {model_name} from HuggingFace...")
            self.loading_status = {
                'stage': 'downloading',
                'message': f'Downloading {model_name}...',
                'progress': start_progress,
                'download_model': model_name
            }
            
            logger.info(f"   Downloading from: {model_id}")
            logger.info(f"   Saving to: {local_dir}")
            
            # Download with snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                force_download=False,
                token=None
            )
            
            self.loading_status['progress'] = end_progress
            self.loading_status['message'] = f'{model_name} downloaded successfully!'
            
            logger.info(f"✅ {model_name} downloaded successfully to {local_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading {model_name}: {str(e)}")
            self.loading_status = {
                'stage': 'error',
                'message': f'Error downloading {model_name}: {str(e)}',
                'progress': 0
            }
            return False
    
    def check_and_download_models(self):
        """Check if models exist locally, download if missing"""
        try:
            if not self.config:
                raise RuntimeError("ModelManager not initialized with config")
            
            os.makedirs(self.config['MODELS_BASE_DIR'], exist_ok=True)
            
            self.loading_status = {
                'stage': 'checking',
                'message': 'Checking models...',
                'progress': 5
            }
            
            # Check OlmOCR model
            olmocr_exists = os.path.exists(self.config['OLMOCR_MODEL_DIR']) and os.path.exists(
                os.path.join(self.config['OLMOCR_MODEL_DIR'], "config.json")
            )
            
            # Check Qwen model
            qwen_exists = os.path.exists(self.config['QWEN_MODEL_DIR']) and os.path.exists(
                os.path.join(self.config['QWEN_MODEL_DIR'], "config.json")
            )
            
            logger.info("=" * 60)
            logger.info("MODEL STATUS CHECK")
            logger.info("=" * 60)
            logger.info(f"OlmOCR-7B-FP4: {'✅ Found' if olmocr_exists else '❌ Missing'}")
            logger.info(f"Qwen3-1.7B: {'✅ Found' if qwen_exists else '❌ Missing'}")
            logger.info("=" * 60)
            
            # Download missing models
            if not olmocr_exists:
                logger.info("📥 OlmOCR model not found, downloading...")
                if not self.download_model_with_progress(
                    self.config['OLMOCR_MODEL_ID'], 
                    self.config['OLMOCR_MODEL_DIR'], 
                    "OlmOCR-7B-FP4", 10, 50
                ):
                    return False
            else:
                logger.info("✅ OlmOCR model found locally")
                self.loading_status['progress'] = 50
            
            if not qwen_exists:
                logger.info("📥 Qwen model not found, downloading...")
                if not self.download_model_with_progress(
                    self.config['QWEN_MODEL_ID'], 
                    self.config['QWEN_MODEL_DIR'], 
                    "Qwen3-1.7B", 50, 90
                ):
                    return False
            else:
                logger.info("✅ Qwen model found locally")
                self.loading_status['progress'] = 90
            
            self.loading_status = {
                'stage': 'ready_to_load',
                'message': 'Models ready, loading...',
                'progress': 90
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in model check: {str(e)}")
            self.loading_status = {
                'stage': 'error',
                'message': f'Error checking models: {str(e)}',
                'progress': 0
            }
            return False
    
    def load_olmocr_model(self):
        """Load the OlmOCR 4-bit quantized model and processor"""
        try:
            if not self.config:
                raise RuntimeError("ModelManager not initialized with config")
            
            self.loading_status = {'stage': 'loading', 'message': 'Loading OlmOCR model...', 'progress': 92}
            logger.info("=" * 60)
            logger.info("Loading OlmOCR-7B-FP4 Model")
            logger.info("=" * 60)
            
            logger.info(f"📂 Loading model from: {self.config['OLMOCR_MODEL_DIR']}")
            
            # Load processor
            self.loading_status = {'stage': 'loading', 'message': 'Loading processor...', 'progress': 93}
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.config['OLMOCR_MODEL_DIR'], trust_remote_code=True)
            logger.info("✅ Processor loaded successfully")
            
            # Load model with device mapping
            self.loading_status = {'stage': 'loading', 'message': 'Loading model to device...', 'progress': 95}
            logger.info("Loading model...")
            
            if torch.cuda.is_available():
                logger.info(f"🎮 CUDA available! GPU: {torch.cuda.get_device_name(0)}")
                self.loading_status = {'stage': 'loading', 'message': f'Loading model on GPU: {torch.cuda.get_device_name(0)}', 'progress': 96}
                
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.config['OLMOCR_MODEL_DIR'],
                    trust_remote_code=True,
                    device_map="auto",
                    dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    max_memory={0: "7GB", "cpu": "16GB"}
                )
            else:
                logger.warning("⚠️  CUDA not available - using CPU (will be SLOW)")
                self.loading_status = {'stage': 'loading', 'message': 'Loading model on CPU (slower)...', 'progress': 96}
                
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.config['OLMOCR_MODEL_DIR'],
                    trust_remote_code=True,
                    device_map="cpu",
                    dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            self.loading_status = {'stage': 'finalizing', 'message': 'Finalizing model setup...', 'progress': 98}
            self.model.eval()
            
            self.loading_status = {'stage': 'ready', 'message': 'Model loaded successfully!', 'progress': 100}
            logger.info("✅ OlmOCR model loaded successfully!")
            logger.info(f"   Device: {next(self.model.parameters()).device}")
            logger.info("=" * 60)
            
        except Exception as e:
            self.loading_status = {'stage': 'error', 'message': f'Error loading model: {str(e)}', 'progress': 0}
            logger.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def load_qwen_model(self):
        """DEPRECATED: Use ensure_qwen_loaded() instead. Load Qwen model for parsing (cached globally)"""
        logger.warning("⚠️ load_qwen_model() is deprecated, use ensure_qwen_loaded() instead")
        return self.ensure_qwen_loaded()
    
    def initialize_models(self):
        """Check/download models but DON'T load them yet (lazy loading)"""
        try:
            # Only check and download if needed, don't load into memory/GPU
            if not self.check_and_download_models():
                logger.error("❌ Failed to download models")
                return False
            
            # Mark as ready but don't load
            self.loading_status = {
                'stage': 'ready',
                'message': 'Models ready (will load on first use)',
                'progress': 100
            }
            logger.info("✅ Models available - will load on demand to save GPU memory")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in model initialization: {str(e)}")
            self.loading_status['stage'] = 'error'
            self.loading_status['message'] = str(e)
            self.loading_status['progress'] = 0
            return False
    
    def _start_olmocr_unload_timer(self):
        """Start timer to auto-unload OlmOCR after inactivity"""
        # Cancel existing timer if any
        if self.olmocr_timer is not None:
            self.olmocr_timer.cancel()
        
        # Start new timer
        def auto_unload():
            if self.olmocr_last_used is not None:
                elapsed = time.time() - self.olmocr_last_used
                if elapsed >= self.unload_timeout and self.model is not None:
                    logger.info(f"⏰ Auto-unloading OlmOCR after {elapsed:.0f}s of inactivity")
                    self.unload_olmocr_model()
        
        self.olmocr_timer = threading.Timer(self.unload_timeout, auto_unload)
        self.olmocr_timer.daemon = True
        self.olmocr_timer.start()
    
    def _start_qwen_unload_timer(self):
        """Start timer to auto-unload Qwen after inactivity"""
        # Cancel existing timer if any
        if self.qwen_timer is not None:
            self.qwen_timer.cancel()
        
        # Start new timer
        def auto_unload():
            if self.qwen_last_used is not None:
                elapsed = time.time() - self.qwen_last_used
                if elapsed >= self.unload_timeout and self.qwen_model is not None:
                    logger.info(f"⏰ Auto-unloading Qwen after {elapsed:.0f}s of inactivity")
                    self.unload_qwen_model()
        
        self.qwen_timer = threading.Timer(self.unload_timeout, auto_unload)
        self.qwen_timer.daemon = True
        self.qwen_timer.start()
    
    def ensure_olmocr_loaded(self):
        """Lazy load OlmOCR model only when needed"""
        if self.model is None or self.processor is None:
            logger.info("🔄 Loading OlmOCR model on-demand...")
            self.load_olmocr_model()
        
        # Update last used timestamp and restart timer
        self.olmocr_last_used = time.time()
        self._start_olmocr_unload_timer()
        
        return self.model, self.processor
    
    def ensure_qwen_loaded(self):
        """Lazy load Qwen model only when needed"""
        if self.qwen_tokenizer is None or self.qwen_model is None:
            if not self.config:
                raise RuntimeError("ModelManager not initialized with config")
            
            logger.info(f"📦 Loading Qwen model from: {self.config['QWEN_MODEL_DIR']}")
            
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.config['QWEN_MODEL_DIR'],
                trust_remote_code=True
            )
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                self.config['QWEN_MODEL_DIR'],
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("✅ Qwen model loaded and cached successfully")
        else:
            logger.info("✅ Using cached Qwen model")
        
        # Update last used timestamp and restart timer
        self.qwen_last_used = time.time()
        self._start_qwen_unload_timer()
        
        return self.qwen_tokenizer, self.qwen_model
    

    def unload_olmocr_model(self):
        """Unload OlmOCR model to free GPU memory"""
        if self.model is not None:
            logger.info("🗑️  Unloading OlmOCR model to free GPU memory...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✅ GPU memory cleared")
            
    def unload_qwen_model(self):
        """Unload Qwen model to free GPU memory"""
        if self.qwen_model is not None:
            logger.info("🗑️  Unloading Qwen model to free GPU memory...")
            del self.qwen_model
            del self.qwen_tokenizer
            self.qwen_model = None
            self.qwen_tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("✅ GPU memory cleared")


# Global model manager instance
model_manager = ModelManager()
