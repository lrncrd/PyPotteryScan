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

# Model directory - Download the quantized model from Google Drive
MODEL_DIR = os.path.join(os.path.dirname(__file__), "olmocr_4bit_model")

def load_model():
    """Load the OlmOCR 4-bit quantized model and processor"""
    global processor, model, loading_status
    try:
        loading_status = {'stage': 'checking', 'message': 'Checking model directory...', 'progress': 10}
        logger.info("=" * 60)
        logger.info("Loading OlmOCR-7B 4-bit Quantized Model")
        logger.info("=" * 60)
        
        # Check if model directory exists
        if not os.path.exists(MODEL_DIR):
            loading_status = {'stage': 'error', 'message': f'Model directory not found: {MODEL_DIR}', 'progress': 0}
            logger.error(f"❌ Model directory not found: {MODEL_DIR}")
            logger.error("📥 Please download the quantized model from Google Drive:")
            logger.error(f"   1. Download 'olmocr-7b-4bit-quantized' folder")
            logger.error(f"   2. Rename to 'olmocr_4bit_model'")
            logger.error(f"   3. Place in: {os.path.dirname(__file__)}")
            logger.error("")
            logger.error("See DOWNLOAD_MODEL.md for detailed instructions")
            raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
        
        # Check if config exists
        loading_status = {'stage': 'checking', 'message': 'Checking model configuration...', 'progress': 20}
        config_file = os.path.join(MODEL_DIR, "config.json")
        if not os.path.exists(config_file):
            loading_status = {'stage': 'error', 'message': 'Model configuration not found', 'progress': 0}
            logger.error(f"❌ Model configuration not found: {config_file}")
            logger.error("The model directory exists but seems incomplete.")
            raise FileNotFoundError(f"Model config not found: {config_file}")
        
        logger.info(f"📂 Loading model from: {MODEL_DIR}")
        
        # Load processor
        loading_status = {'stage': 'loading', 'message': 'Loading processor...', 'progress': 30}
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
        logger.info("✅ Processor loaded successfully")
        
        # Load model with device mapping
        loading_status = {'stage': 'loading', 'message': 'Loading 4-bit quantized model (this may take a few moments)...', 'progress': 50}
        logger.info("Loading 4-bit quantized model...")
        logger.info("This may take a few moments...")
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            logger.info(f"🎮 CUDA available! GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   CUDA Version: {torch.version.cuda}")
            loading_status = {'stage': 'loading', 'message': f'Loading model on GPU: {torch.cuda.get_device_name(0)}', 'progress': 60}
            
            # Configurazione ottimizzata per RTX 3060 8GB
            # Un modello 7B a 4-bit occupa ~3.5-4GB, quindi dovrebbe stare in VRAM
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_DIR,
                trust_remote_code=True,
                device_map="auto",
                dtype=torch.float16,  # Usa dtype invece di torch_dtype
                low_cpu_mem_usage=True,  # Riduce l'uso di RAM durante il caricamento
                max_memory={0: "7GB", "cpu": "16GB"}  # Lascia 1GB libero per CUDA
            )
        else:
            logger.warning("⚠️  CUDA not available - using CPU (will be SLOW)")
            logger.warning("For best performance, use Windows PC with RTX 3060")
            loading_status = {'stage': 'loading', 'message': 'Loading model on CPU (slower)...', 'progress': 60}
            
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_DIR,
                trust_remote_code=True,
                device_map="cpu",
                dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        
        loading_status = {'stage': 'finalizing', 'message': 'Finalizing model setup...', 'progress': 90}
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
    """Load model in background thread"""
    load_model()

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