#!/usr/bin/env python3
"""
OCR Server using OlmOCR-7B (4-bit quantized) for advanced text recognition
Optimized for ceramic plate artifacts with mixed Italian/English text
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import io
import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import logging
import webbrowser
import threading
import time
import zipfile
import csv
from datetime import datetime
import pandas as pd
import json
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and processor
processor = None
model = None
loading_status = {
    'stage': 'starting',
    'message': 'Initializing server...',
    'progress': 0
}

# Global variables for Qwen parser model (cached)
qwen_tokenizer = None
qwen_model = None

# Global parsing progress tracker
parsing_status = {
    'active': False,
    'current': 0,
    'total': 0,
    'current_line': ''
}

# Model directories
MODELS_BASE_DIR = os.path.join(os.path.dirname(__file__), "models")
OLMOCR_MODEL_DIR = os.path.join(MODELS_BASE_DIR, "olmocr-7b-fp4")
QWEN_MODEL_DIR = os.path.join(MODELS_BASE_DIR, "qwen3-1.7b")

# HuggingFace model IDs
OLMOCR_MODEL_ID = "lrncrd/olmOCR-7B-FP4"
QWEN_MODEL_ID = "Qwen/Qwen3-1.7B"

def download_model_with_progress(model_id, local_dir, model_name, start_progress=10, end_progress=50):
    """Download model from HuggingFace with progress tracking"""
    global loading_status
    
    try:
        logger.info(f"📥 Downloading {model_name} from HuggingFace...")
        loading_status = {
            'stage': 'downloading',
            'message': f'Downloading {model_name}...',
            'progress': start_progress,
            'download_model': model_name
        }
        
        logger.info(f"   Downloading from: {model_id}")
        logger.info(f"   Saving to: {local_dir}")
        
        # Create a thread to simulate progress during download
        # (HuggingFace snapshot_download doesn't provide real-time progress)
        import threading
        download_complete = threading.Event()
        
        def simulate_progress():
            """Simulate download progress"""
            current = start_progress
            while not download_complete.is_set() and current < end_progress - 5:
                time.sleep(2)  # Update every 2 seconds
                current += 2
                loading_status['progress'] = min(current, end_progress - 5)
                loading_status['message'] = f'Downloading {model_name}... {min(current, end_progress - 5)}%'
                logger.info(f"   Download progress: {min(current, end_progress - 5)}%")
        
        # Start progress simulation
        progress_thread = threading.Thread(target=simulate_progress, daemon=True)
        progress_thread.start()
        
        # Download with snapshot_download
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            force_download=False,
            token=None  # Use public models
        )
        
        # Stop progress simulation
        download_complete.set()
        
        # Set to end progress
        loading_status['progress'] = end_progress
        loading_status['message'] = f'{model_name} downloaded successfully!'
        
        logger.info(f"✅ {model_name} downloaded successfully to {local_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        loading_status = {
            'stage': 'error',
            'message': f'Error downloading {model_name}: {str(e)}',
            'progress': 0
        }
        return False

def check_and_download_models():
    """Check if models exist locally, download if missing"""
    global loading_status
    
    try:
        # Create models base directory if it doesn't exist
        os.makedirs(MODELS_BASE_DIR, exist_ok=True)
        
        loading_status = {
            'stage': 'checking',
            'message': 'Checking models...',
            'progress': 5
        }
        
        # Check OlmOCR model
        olmocr_exists = os.path.exists(OLMOCR_MODEL_DIR) and os.path.exists(
            os.path.join(OLMOCR_MODEL_DIR, "config.json")
        )
        
        # Check Qwen model
        qwen_exists = os.path.exists(QWEN_MODEL_DIR) and os.path.exists(
            os.path.join(QWEN_MODEL_DIR, "config.json")
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
            loading_status = {
                'stage': 'downloading',
                'message': 'Downloading OlmOCR-7B-FP4 model...',
                'progress': 10
            }
            if not download_model_with_progress(OLMOCR_MODEL_ID, OLMOCR_MODEL_DIR, "OlmOCR-7B-FP4", 10, 50):
                return False
        else:
            logger.info("✅ OlmOCR model found locally")
            loading_status['progress'] = 50
        
        if not qwen_exists:
            logger.info("📥 Qwen model not found, downloading...")
            loading_status = {
                'stage': 'downloading',
                'message': 'Downloading Qwen3-1.7B model...',
                'progress': 50
            }
            if not download_model_with_progress(QWEN_MODEL_ID, QWEN_MODEL_DIR, "Qwen3-1.7B", 50, 90):
                return False
        else:
            logger.info("✅ Qwen model found locally")
            loading_status['progress'] = 90
        
        loading_status = {
            'stage': 'ready_to_load',
            'message': 'Models ready, loading...',
            'progress': 90
        }
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in model check: {str(e)}")
        loading_status = {
            'stage': 'error',
            'message': f'Error checking models: {str(e)}',
            'progress': 0
        }
        return False

def load_model():
    """Load the OlmOCR 4-bit quantized model and processor"""
    global processor, model, loading_status
    try:
        loading_status = {'stage': 'loading', 'message': 'Loading OlmOCR model...', 'progress': 92}
        logger.info("=" * 60)
        logger.info("Loading OlmOCR-7B-FP4 Model")
        logger.info("=" * 60)
        
        logger.info(f"📂 Loading model from: {OLMOCR_MODEL_DIR}")
        
        # Load processor
        loading_status = {'stage': 'loading', 'message': 'Loading processor...', 'progress': 93}
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(OLMOCR_MODEL_DIR, trust_remote_code=True)
        logger.info("✅ Processor loaded successfully")
        
        # Load model with device mapping
        loading_status = {'stage': 'loading', 'message': 'Loading model to device...', 'progress': 95}
        logger.info("Loading model...")
        logger.info("This may take a few moments...")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            logger.info(f"🎮 CUDA available! GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   CUDA Version: {torch.version.cuda}")
            loading_status = {'stage': 'loading', 'message': f'Loading model on GPU: {torch.cuda.get_device_name(0)}', 'progress': 96}
            
            # Configurazione ottimizzata per RTX 3060 8GB
            # Un modello 7B a 4-bit occupa ~3.5-4GB, quindi dovrebbe stare in VRAM
            model = AutoModelForImageTextToText.from_pretrained(
                OLMOCR_MODEL_DIR,
                trust_remote_code=True,
                device_map="auto",
                dtype=torch.float16,  # Usa dtype invece di torch_dtype
                low_cpu_mem_usage=True,  # Riduce l'uso di RAM durante il caricamento
                max_memory={0: "7GB", "cpu": "16GB"}  # Lascia 1GB libero per CUDA
            )
        else:
            logger.warning("⚠️  CUDA not available - using CPU (will be SLOW)")
            logger.warning("For best performance, use Windows PC with RTX 3060")
            loading_status = {'stage': 'loading', 'message': 'Loading model on CPU (slower)...', 'progress': 96}
            
            model = AutoModelForImageTextToText.from_pretrained(
                OLMOCR_MODEL_DIR,
                trust_remote_code=True,
                device_map="cpu",
                dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        
        loading_status = {'stage': 'finalizing', 'message': 'Finalizing model setup...', 'progress': 98}
        model.eval()  # Set to evaluation mode
        
        loading_status = {'stage': 'ready', 'message': 'Model loaded successfully!', 'progress': 100}
        logger.info("✅ OlmOCR model loaded successfully!")
        logger.info(f"   Device: {next(model.parameters()).device}")
        logger.info("=" * 60)
        
    except Exception as e:
        loading_status = {'stage': 'error', 'message': f'Error loading model: {str(e)}', 'progress': 0}
        logger.error(f"❌ Error loading model: {str(e)}")
        raise

def preprocess_image(image):
    """Preprocess image for OCR"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        logger.warning(f"Error in preprocessing: {str(e)}, using original image")
        return image.convert('RGB')

def load_qwen_model():
    """Load Qwen model for parsing (cached globally)"""
    global qwen_tokenizer, qwen_model
    
    if qwen_tokenizer is not None and qwen_model is not None:
        logger.info("✅ Using cached Qwen model")
        return qwen_tokenizer, qwen_model
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    logger.info(f"📦 Loading Qwen model from: {QWEN_MODEL_DIR}")
    
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_DIR,
        trust_remote_code=True
    )
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_DIR,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    logger.info("✅ Qwen model loaded and cached successfully")
    
    return qwen_tokenizer, qwen_model

def process_image_ocr(image_data):
    """Process image with OlmOCR and return recognized text"""
    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        logger.info(f"📷 Image size: {image.size}")
        
        # Preprocess image
        image = preprocess_image(image)
        
        # OlmOCR/Qwen2.5-VL usa un formato chat con messaggi
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract all text from this image:"},
                ],
            }
        ]
        
        # Prepara il prompt usando apply_chat_template
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move to model device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # Generate text
        logger.info("🔍 Processing with OlmOCR...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                #cache=True  # Abilita caching per velocizzare la generazione
            )
        
        # Decodifica solo i nuovi token generati
        input_len = inputs.input_ids.shape[1]
        generated_ids = [output_id[input_len:] for output_id in output_ids]
        generated_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        
        # Forza tutto su una sola riga, rimuovendo a capo
        generated_text = generated_text.replace('\n', ' ').replace('\r', ' ')
        generated_text = ' '.join(generated_text.split())  # Normalizza spazi multipli
        
        logger.info(f"✨ Result: '{generated_text}'")
        return generated_text.strip() if generated_text.strip() else "No text detected"
        
    except Exception as e:
        logger.error(f"❌ Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ocr_engine': 'OlmOCR-7B',
        'model': 'allenai/olmOCR-7B-0825-FP8 (4-bit quantized)',
        'quantization': '4-bit',
        'model_loaded': model is not None and processor is not None,
        'cuda_available': torch.cuda.is_available(),
        'device': str(next(model.parameters()).device) if model is not None else 'unknown'
    })

@app.route('/loading_status', methods=['GET'])
def get_loading_status():
    """Get current loading status for splash screen"""
    global loading_status
    return jsonify(loading_status)

@app.route('/parsing_status', methods=['GET'])
def get_parsing_status():
    """Get current parsing progress"""
    global parsing_status
    return jsonify(parsing_status)

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    """OCR endpoint to process image data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process the image
        result_text = process_image_ocr(data['image'])
        
        return jsonify({
            'success': True,
            'text': result_text
        })
        
    except Exception as e:
        logger.error(f"Error in OCR endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch_ocr', methods=['POST'])
def batch_ocr_endpoint():
    """Batch OCR endpoint to process multiple images"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'images' not in data or not isinstance(data['images'], list):
            return jsonify({'error': 'No images array provided'}), 400
        
        results = []
        for i, image_data in enumerate(data['images']):
            try:
                text = process_image_ocr(image_data)
                results.append({
                    'index': i,
                    'success': True,
                    'text': text
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in batch OCR endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/export', methods=['POST'])
def export_endpoint():
    """Export endpoint to generate ZIP with renamed images and CSV"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract data
        images_data = data.get('images', [])
        annotations = data.get('annotations', {})
        ocr_results = data.get('ocrResults', {})
        corrections = data.get('corrections', {})
        prefix = data.get('prefix', 'ceramic')
        
        # Create in-memory ZIP file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Create CSV data
            csv_buffer = io.StringIO()
            csv_writer = csv.writer(csv_buffer)
            csv_writer.writerow(['filename', 'original_filename', 'table_name', 'context', 'notes', 'ocr_result', 'ocr_corrected'])
            
            # Process each image
            for idx, img_data in enumerate(images_data):
                try:
                    img_name = img_data.get('name', f'image_{idx}')
                    img_base64 = img_data.get('data', '')
                    
                    # Decode image
                    if img_base64.startswith('data:image'):
                        img_base64 = img_base64.split(',')[1]
                    
                    img_bytes = base64.b64decode(img_base64)
                    
                    # Generate new filename
                    new_filename = f"{prefix}_{str(idx + 1).padStart(3, '0')}.jpg"
                    
                    # Add image to ZIP
                    zip_file.writestr(f"images/{new_filename}", img_bytes)
                    
                    # Get annotation data
                    annotation = annotations.get(img_name, {})
                    ocr_result = ocr_results.get(img_name, '')
                    corrected = corrections.get(img_name, ocr_result)
                    
                    # Add to CSV
                    csv_writer.writerow([
                        new_filename,
                        img_name,
                        annotation.get('tableName', ''),
                        annotation.get('context', ''),
                        annotation.get('notes', ''),
                        ocr_result,
                        corrected
                    ])
                    
                except Exception as e:
                    logger.error(f"Error processing image {idx}: {str(e)}")
            
            # Add CSV to ZIP
            zip_file.writestr(f"{prefix}_data.csv", csv_buffer.getvalue())
        
        # Prepare for download
        zip_buffer.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'{prefix}_export_{timestamp}.zip'
        )
        
    except Exception as e:
        logger.error(f"Error in export endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/parse_structured', methods=['POST'])
def parse_structured_endpoint():
    """Parse OCR data using Qwen3 model with few-shot examples"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        ocr_lines = data.get('ocrLines', [])
        fewshot_examples = data.get('fewshotExamples', [])
        filenames = data.get('filenames', [])  # Get filenames array
        use_guided = data.get('useGuided', True)
        
        if not ocr_lines:
            return jsonify({'error': 'No OCR lines provided'}), 400
        
        # Ensure filenames array matches ocr_lines length
        if len(filenames) != len(ocr_lines):
            filenames = [''] * len(ocr_lines)  # Fill with empty strings if missing
        
        logger.info(f"🔍 Starting structured parsing for {len(ocr_lines)} lines")
        logger.info(f"   Mode: {'Guided (with few-shot)' if use_guided else 'Free (no examples)'}")
        logger.info(f"   Few-shot examples: {len(fewshot_examples) // 2}")
        
        # Initialize parsing status
        global parsing_status
        parsing_status = {
            'active': True,
            'current': 0,
            'total': len(ocr_lines),
            'current_line': ''
        }
        
        # System prompt
        system_base = (
            "You are an assistant specialised in parsing short archaeological OCR lines. "
            "Return only one JSON object with these possible fields: "
            "Inventory, Site, Year, US, Area, Cut, Sector, Notes. "
            "If something is missing, set it to null. Avoid text outside JSON."
        )
        
        # Load Qwen model (uses global cache)
        logger.info(f"📦 Loading/checking Qwen model...")
        tokenizer, qwen_model = load_qwen_model()
        logger.info("✅ Qwen model ready")
        
        results = []
        
        # Use the already loaded/cached Qwen model
        tokenizer = qwen_tokenizer
        
        results = []
        for i, line in enumerate(ocr_lines):
            try:
                # Update parsing status
                parsing_status['current'] = i + 1
                parsing_status['current_line'] = line
                
                filename = filenames[i] if i < len(filenames) else ''
                logger.info(f"[{i+1}/{len(ocr_lines)}] File: {filename} | Parsing: {line}")
                
                # Build messages
                logger.info(f"   📝 Building prompt with {len(fewshot_examples) // 2 if use_guided else 0} examples...")
                system = {"role": "system", "content": system_base}
                user = {"role": "user", "content": f"Parse this OCR line and return JSON only: {line}"}
                
                if use_guided and fewshot_examples:
                    messages = [system] + fewshot_examples + [user]
                else:
                    messages = [system, user]
                
                # Apply chat template
                logger.info(f"   🔧 Applying chat template...")
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  # DISABLE thinking mode for direct responses
                )
                logger.info(f"   📊 Tokenizing input (length: {len(prompt)} chars)...")
                inputs = tokenizer([prompt], return_tensors="pt").to(qwen_model.device)
                
                # Generate with temperature 0 for deterministic output
                logger.info(f"   🤖 Generating response (this may take 10-30 seconds)...")
                import time
                start_time = time.time()
                
                generated_ids = qwen_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False
                )
                
                elapsed = time.time() - start_time
                logger.info(f"   ⏱️  Generation completed in {elapsed:.1f}s")
                
                logger.info(f"   📤 Decoding response...")
                content = tokenizer.decode(
                    generated_ids[0][len(inputs.input_ids[0]):], 
                    skip_special_tokens=True
                )
                
                # Parse JSON
                logger.info(f"   🔍 Parsing JSON from response...")
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    logger.warning(f"   ⚠️  Direct JSON parse failed, trying regex extraction...")
                    import re
                    json_match = re.search(r'\{[^}]+\}', content)
                    if json_match:
                        parsed = json.loads(json_match.group())
                    else:
                        parsed = {"raw_response": content}
                
                parsed["_ocr_original"] = line
                parsed["_filename"] = filename  # Add filename to results
                results.append(parsed)
                logger.info(f"   ✓ Result: {json.dumps(parsed, ensure_ascii=False)}")
                logger.info("")  # Empty line for readability
                
            except Exception as e:
                logger.error(f"   ⚠️ Error on row {i}: {e}")
                filename = filenames[i] if i < len(filenames) else ''
                results.append({"_ocr_original": line, "_filename": filename, "error": str(e)})
        
        # Reset parsing status
        parsing_status['active'] = False
        parsing_status['current'] = parsing_status['total']
        
        # Create Excel file in memory
        logger.info("📊 Creating Excel file...")
        df = pd.json_normalize(results)
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Parsed Data')
        
        excel_buffer.seek(0)
        
        logger.info("✅ Parsing complete! Sending Excel file...")
        
        # Return Excel file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return send_file(
            excel_buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'parsed_output_{timestamp}.xlsx'
        )
        
    except Exception as e:
        logger.error(f"❌ Error in parse_structured endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def open_browser(port):
    """Apre il browser IMMEDIATAMENTE senza aspettare"""
    time.sleep(1)  # Breve pausa per far partire il server
    # Apri direttamente il file HTML dell'interfaccia
    html_path = os.path.join(os.path.dirname(__file__), "ceramic-workflow-v2.html")
    if os.path.exists(html_path):
        html_url = f"file:///{html_path.replace(chr(92), '/')}"  # Converte backslash in forward slash
        logger.info(f"🌐 Apertura interfaccia OCR: {html_url}")
        webbrowser.open(html_url)
    else:
        # Fallback all'endpoint di health
        url = f"http://localhost:{port}/health"
        logger.info(f"🌐 Apertura browser: {url}")
        webbrowser.open(url)

def load_model_async():
    """Check/download models and load them in background thread"""
    try:
        # First check and download models if needed
        if not check_and_download_models():
            logger.error("❌ Failed to download models")
            return
        
        # Then load the OlmOCR model
        load_model()
    except Exception as e:
        logger.error(f"❌ Error in model loading: {str(e)}")
        loading_status['stage'] = 'error'
        loading_status['message'] = str(e)
        loading_status['progress'] = 0

if __name__ == '__main__':
    try:
        port = 5001
        
        # Start Flask server first
        logger.info(f"🚀 Starting OCR server on http://localhost:{port}")
        logger.info(f"📍 Health check: http://localhost:{port}/health")
        logger.info(f"📍 Loading status: http://localhost:{port}/loading_status")
        logger.info(f"📍 OCR endpoint: http://localhost:{port}/ocr")
        
        # Open browser immediately
        threading.Thread(target=open_browser, args=(port,), daemon=True).start()
        
        # Load model in background
        threading.Thread(target=load_model_async, daemon=False).start()
        
        # Start server (this will run while model loads)
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        exit(1)