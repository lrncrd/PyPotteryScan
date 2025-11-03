"""
Routes for PyPotteryScan Flask Application
"""
import base64
import io
import os
import json
import time
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, send_file
from PIL import Image
import torch
import pandas as pd
import logging

from app.model_manager import model_manager

logger = logging.getLogger(__name__)

# Create blueprints
main_bp = Blueprint('main', __name__)
ocr_bp = Blueprint('ocr', __name__, url_prefix='/api')
parser_bp = Blueprint('parser', __name__, url_prefix='/api')


# ==================
# MAIN ROUTES
# ==================

@main_bp.route('/')
def index():
    """Render main interface"""
    return render_template('index.html')


@main_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ocr_engine': 'OlmOCR-7B',
        'model': 'allenai/olmOCR-7B-0825-FP8 (4-bit quantized)',
        'quantization': '4-bit',
        'model_loaded': model_manager.model is not None and model_manager.processor is not None,
        'cuda_available': torch.cuda.is_available(),
        'device': str(next(model_manager.model.parameters()).device) if model_manager.model is not None else 'unknown'
    })


@main_bp.route('/loading_status', methods=['GET'])
def get_loading_status():
    """Get current loading status for splash screen"""
    return jsonify(model_manager.get_loading_status())


@main_bp.route('/parsing_status', methods=['GET'])
def get_parsing_status():
    """Get current parsing progress"""
    return jsonify(model_manager.get_parsing_status())


# ==================
# OCR ROUTES
# ==================

def preprocess_image(image):
    """Preprocess image for OCR"""
    try:
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
        
        # OlmOCR uses chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract all text from this image:"},
                ],
            }
        ]
        
        # Prepare prompt
        inputs = model_manager.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move to model device
        device = next(model_manager.model.parameters()).device
        inputs = inputs.to(device)
        
        # Generate text
        logger.info("🔍 Processing with OlmOCR...")
        with torch.no_grad():
            output_ids = model_manager.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=model_manager.processor.tokenizer.pad_token_id,
            )
        
        # Decode generated tokens
        input_len = inputs.input_ids.shape[1]
        generated_ids = [output_id[input_len:] for output_id in output_ids]
        generated_text = model_manager.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        
        # Force single line
        generated_text = generated_text.replace('\n', ' ').replace('\r', ' ')
        generated_text = ' '.join(generated_text.split())
        
        logger.info(f"✨ Result: '{generated_text}'")
        return generated_text.strip() if generated_text.strip() else "No text detected"
        
    except Exception as e:
        logger.error(f"❌ Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


@ocr_bp.route('/ocr', methods=['POST'])
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


@ocr_bp.route('/batch_ocr', methods=['POST'])
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


# ==================
# PARSER ROUTES
# ==================

@parser_bp.route('/parse_structured', methods=['POST'])
def parse_structured_endpoint():
    """Parse OCR data using Qwen3 model with few-shot examples"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        ocr_lines = data.get('ocrLines', [])
        fewshot_examples = data.get('fewshotExamples', [])
        filenames = data.get('filenames', [])
        use_guided = data.get('useGuided', True)
        
        if not ocr_lines:
            return jsonify({'error': 'No OCR lines provided'}), 400
        
        # Ensure filenames array matches ocr_lines length
        if len(filenames) != len(ocr_lines):
            filenames = [''] * len(ocr_lines)
        
        logger.info(f"🔍 Starting structured parsing for {len(ocr_lines)} lines")
        logger.info(f"   Mode: {'Guided (with few-shot)' if use_guided else 'Free (no examples)'}")
        logger.info(f"   Few-shot examples: {len(fewshot_examples) // 2}")
        
        # Initialize parsing status
        model_manager.parsing_status = {
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
        
        # Load Qwen model
        logger.info(f"📦 Loading/checking Qwen model...")
        tokenizer, qwen_model = model_manager.load_qwen_model()
        logger.info("✅ Qwen model ready")
        
        results = []
        
        for i, line in enumerate(ocr_lines):
            try:
                # Update parsing status
                model_manager.parsing_status['current'] = i + 1
                model_manager.parsing_status['current_line'] = line
                
                filename = filenames[i] if i < len(filenames) else ''
                logger.info(f"[{i+1}/{len(ocr_lines)}] File: {filename} | Parsing: {line}")
                
                # Build messages
                system = {"role": "system", "content": system_base}
                user = {"role": "user", "content": f"Parse this OCR line and return JSON only: {line}"}
                
                if use_guided and fewshot_examples:
                    messages = [system] + fewshot_examples + [user]
                else:
                    messages = [system, user]
                
                # Apply chat template
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                inputs = tokenizer([prompt], return_tensors="pt").to(qwen_model.device)
                
                # Generate
                start_time = time.time()
                generated_ids = qwen_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False
                )
                elapsed = time.time() - start_time
                logger.info(f"   ⏱️  Generation completed in {elapsed:.1f}s")
                
                # Decode
                content = tokenizer.decode(
                    generated_ids[0][len(inputs.input_ids[0]):], 
                    skip_special_tokens=True
                )
                
                # Parse JSON
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    import re
                    json_match = re.search(r'\{[^}]+\}', content)
                    if json_match:
                        parsed = json.loads(json_match.group())
                    else:
                        parsed = {"raw_response": content}
                
                parsed["_ocr_original"] = line
                parsed["_filename"] = filename
                results.append(parsed)
                logger.info(f"   ✓ Result: {json.dumps(parsed, ensure_ascii=False)}")
                
            except Exception as e:
                logger.error(f"   ⚠️ Error on row {i}: {e}")
                filename = filenames[i] if i < len(filenames) else ''
                results.append({"_ocr_original": line, "_filename": filename, "error": str(e)})
        
        # Reset parsing status
        model_manager.parsing_status['active'] = False
        model_manager.parsing_status['current'] = model_manager.parsing_status['total']
        
        # Create Excel file
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
