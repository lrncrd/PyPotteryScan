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
        
        # Model selection state
        self.selected_model = None  # 'FP4' or 'FP8'
        self.needs_model_selection = False  # True if user must choose model
        
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
    
    def get_available_models(self):
        """Return list of available OCR models for this hardware"""
        models = []
        
        # FP4 only available on NVIDIA GPU with CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            models.append({
                'id': 'FP4',
                'name': 'OlmOCR 4-bit (FP4)',
                'size_gb': 5,
                'performance': 'Good',
                'description': 'Lighter model, requires NVIDIA GPU',
                'available': True
            })
        else:
            models.append({
                'id': 'FP4',
                'name': 'OlmOCR 4-bit (FP4)',
                'size_gb': 5,
                'performance': 'Good',
                'description': 'Requires NVIDIA GPU (not available)',
                'available': False
            })
        
        # FP8 available on any system
        models.append({
            'id': 'FP8',
            'name': 'OlmOCR 8-bit (FP8)',
            'size_gb': 10,
            'performance': 'Better',
            'description': 'Higher quality, works on any GPU/CPU',
            'available': True,
            'recommended': True
        })
        
        return models
    
    def get_selected_model(self):
        """Get currently selected model from persistence file"""
        if self.selected_model:
            return self.selected_model
        
        if not self.config:
            return None
        
        selection_file = self.config.get('SELECTED_MODEL_FILE')
        if selection_file and os.path.exists(selection_file):
            try:
                with open(selection_file, 'r') as f:
                    model_id = f.read().strip()
                    if model_id in ['FP4', 'FP8']:
                        self.selected_model = model_id
                        return model_id
            except Exception as e:
                logger.warning(f"Could not read selected model file: {e}")
        
        return None
    
    def set_selected_model(self, model_id):
        """Save model selection to persistence file"""
        if model_id not in ['FP4', 'FP8']:
            raise ValueError(f"Invalid model ID: {model_id}")
        
        # Validate FP4 availability
        if model_id == 'FP4' and not torch.cuda.is_available():
            raise ValueError("FP4 requires NVIDIA GPU with CUDA")
        
        self.selected_model = model_id
        
        if self.config:
            selection_file = self.config.get('SELECTED_MODEL_FILE')
            if selection_file:
                os.makedirs(os.path.dirname(selection_file), exist_ok=True)
                with open(selection_file, 'w') as f:
                    f.write(model_id)
                logger.info(f"✅ Model selection saved: {model_id}")
        
        return True
    
    def get_olmocr_model_config(self):
        """Get model ID and directory for currently selected OlmOCR model"""
        model_id = self.get_selected_model()
        if not model_id:
            raise RuntimeError("No OlmOCR model selected")
        
        if model_id == 'FP4':
            return {
                'model_id': self.config['OLMOCR_FP4_MODEL_ID'],
                'model_dir': self.config['OLMOCR_FP4_MODEL_DIR'],
                'name': 'OlmOCR-7B-FP4'
            }
        else:  # FP8
            return {
                'model_id': self.config['OLMOCR_FP8_MODEL_ID'],
                'model_dir': self.config['OLMOCR_FP8_MODEL_DIR'],
                'name': 'OlmOCR-7B-FP8'
            }
    
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
        """Check if models exist locally, download if missing. Requires model selection first."""
        try:
            if not self.config:
                raise RuntimeError("ModelManager not initialized with config")
            
            os.makedirs(self.config['MODELS_BASE_DIR'], exist_ok=True)
            
            self.loading_status = {
                'stage': 'checking',
                'message': 'Checking models...',
                'progress': 5
            }
            
            # Check if model is selected
            selected = self.get_selected_model()
            
            # Check which OlmOCR models exist
            fp4_exists = os.path.exists(self.config['OLMOCR_FP4_MODEL_DIR']) and os.path.exists(
                os.path.join(self.config['OLMOCR_FP4_MODEL_DIR'], "config.json")
            )
            fp8_exists = os.path.exists(self.config['OLMOCR_FP8_MODEL_DIR']) and os.path.exists(
                os.path.join(self.config['OLMOCR_FP8_MODEL_DIR'], "config.json")
            )
            
            # Check Qwen model
            qwen_exists = os.path.exists(self.config['QWEN_MODEL_DIR']) and os.path.exists(
                os.path.join(self.config['QWEN_MODEL_DIR'], "config.json")
            )
            
            logger.info("=" * 60)
            logger.info("MODEL STATUS CHECK")
            logger.info("=" * 60)
            logger.info(f"OlmOCR-7B-FP4: {'✅ Found' if fp4_exists else '❌ Missing'}")
            logger.info(f"OlmOCR-7B-FP8: {'✅ Found' if fp8_exists else '❌ Missing'}")
            logger.info(f"Qwen3-1.7B: {'✅ Found' if qwen_exists else '❌ Missing'}")
            logger.info(f"Selected model: {selected or 'None'}")
            logger.info("=" * 60)
            
            # If no model selected and none exist, need user selection
            if not selected:
                # Auto-select if one already exists
                if fp4_exists and not fp8_exists:
                    self.set_selected_model('FP4')
                    selected = 'FP4'
                elif fp8_exists and not fp4_exists:
                    self.set_selected_model('FP8')
                    selected = 'FP8'
                elif fp4_exists and fp8_exists:
                    # Both exist, prefer FP8 (better performance)
                    self.set_selected_model('FP8')
                    selected = 'FP8'
                else:
                    # No model exists, need user selection
                    self.needs_model_selection = True
                    self.loading_status = {
                        'stage': 'model_selection',
                        'message': 'Please select an OCR model to download',
                        'progress': 5
                    }
                    logger.info("⏸️  Waiting for model selection...")
                    return 'needs_selection'
            
            # Download selected OlmOCR model if missing
            olmocr_config = self.get_olmocr_model_config()
            olmocr_exists = os.path.exists(olmocr_config['model_dir']) and os.path.exists(
                os.path.join(olmocr_config['model_dir'], "config.json")
            )
            
            if not olmocr_exists:
                logger.info(f"📥 {olmocr_config['name']} not found, downloading...")
                if not self.download_model_with_progress(
                    olmocr_config['model_id'], 
                    olmocr_config['model_dir'], 
                    olmocr_config['name'], 10, 50
                ):
                    return False
            else:
                logger.info(f"✅ {olmocr_config['name']} found locally")
                self.loading_status['progress'] = 50
            
            # Download Qwen model if missing
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
        """Load the selected OlmOCR model (FP4 or FP8) and processor"""
        try:
            if not self.config:
                raise RuntimeError("ModelManager not initialized with config")
            
            # Get selected model configuration
            olmocr_config = self.get_olmocr_model_config()
            model_dir = olmocr_config['model_dir']
            model_name = olmocr_config['name']
            
            self.loading_status = {'stage': 'loading', 'message': f'Loading {model_name}...', 'progress': 92}
            logger.info("=" * 60)
            logger.info(f"Loading {model_name}")
            logger.info("=" * 60)
            
            logger.info(f"📂 Loading model from: {model_dir}")
            
            # Load processor
            self.loading_status = {'stage': 'loading', 'message': 'Loading processor...', 'progress': 93}
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
            logger.info("✅ Processor loaded successfully")
            
            # Load model with device mapping
            self.loading_status = {'stage': 'loading', 'message': 'Loading model to device...', 'progress': 95}
            logger.info("Loading model...")
            
            if torch.cuda.is_available():
                logger.info(f"🎮 CUDA available! GPU: {torch.cuda.get_device_name(0)}")
                self.loading_status = {'stage': 'loading', 'message': f'Loading model on GPU: {torch.cuda.get_device_name(0)}', 'progress': 96}
                
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    max_memory={0: "10GB", "cpu": "16GB"}
                )
            else:
                logger.warning("⚠️  CUDA not available - using CPU (will be SLOW)")
                self.loading_status = {'stage': 'loading', 'message': 'Loading model on CPU (slower)...', 'progress': 96}
                
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            self.loading_status = {'stage': 'finalizing', 'message': 'Finalizing model setup...', 'progress': 98}
            self.model.eval()
            
            self.loading_status = {'stage': 'ready', 'message': 'Model loaded successfully!', 'progress': 100}
            logger.info(f"✅ {model_name} loaded successfully!")
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
            result = self.check_and_download_models()
            
            # Handle different return values
            if result == 'needs_selection':
                # Stay in model_selection stage, don't change status
                logger.info("⏸️  Waiting for user to select model...")
                return True  # Not an error, just needs user input
            elif not result:
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
    
    def parse_with_fewshot(self, ocr_text: str, fewshot_examples: list, fields: list) -> dict:
        """
        Parse OCR text using Qwen model with few-shot examples
        
        Args:
            ocr_text: OCR text to parse
            fewshot_examples: List of few-shot example dicts with 'ocr_text' and 'parsed_data'
            fields: List of field names to extract
            
        Returns:
            Dictionary with parsed field values
        """
        import json
        import re
        
        # Ensure Qwen model is loaded
        tokenizer, qwen_model = self.ensure_qwen_loaded()
        
        # System prompt
        fields_str = ', '.join(fields)
        system_content = (
            f"You are an assistant specialised in parsing short archaeological OCR lines. "
            f"Return only one JSON object with these possible fields: {fields_str}. "
            f"If something is missing, set it to null. Avoid text outside JSON."
        )
        system = {"role": "system", "content": system_content}
        
        # Build few-shot messages from examples
        fewshot_messages = []
        for example in fewshot_examples:
            user_msg = {
                "role": "user",
                "content": f"Parse this OCR line and return JSON only: {example['ocr_text']}"
            }
            assistant_msg = {
                "role": "assistant",
                "content": json.dumps(example['parsed_data'], ensure_ascii=False)
            }
            fewshot_messages.extend([user_msg, assistant_msg])
        
        # User query
        user = {
            "role": "user",
            "content": f"Parse this OCR line and return JSON only: {ocr_text}"
        }
        
        # Complete message sequence
        messages = [system] + fewshot_messages + [user]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer([prompt], return_tensors="pt").to(qwen_model.device)
        
        # Generate
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False
        )
        
        # Decode
        content = tokenizer.decode(
            generated_ids[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        
        # Parse JSON
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                # Return empty dict if parsing fails
                parsed = {field: "" for field in fields}
        
        # Ensure all fields are present
        for field in fields:
            if field not in parsed:
                parsed[field] = ""
        
        return parsed


# Global model manager instance
model_manager = ModelManager()
