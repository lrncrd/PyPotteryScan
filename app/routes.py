"""
Routes for PyPotteryScan Flask Application
"""
import base64
import io
import os
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from PIL import Image
import torch
import pandas as pd
import logging

from app.model_manager import model_manager
from app.project_manager import ProjectManager
from app.config import Config

logger = logging.getLogger(__name__)

# Create blueprints
main_bp = Blueprint('main', __name__)
ocr_bp = Blueprint('ocr', __name__, url_prefix='/api')
parser_bp = Blueprint('parser', __name__, url_prefix='/api')
project_bp = Blueprint('project', __name__, url_prefix='/api/project')

# Initialize project manager with absolute path
project_manager = ProjectManager(projects_root=Config.PROJECTS_DIR)


# ==================
# MAIN ROUTES
# ==================

@main_bp.route('/')
def index():
    """Render main interface"""
    return render_template('index.html', version='0.2.0')


@main_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Check if model is loaded WITHOUT forcing it to load
    model_loaded = model_manager.model is not None and model_manager.processor is not None
    device = 'not loaded yet'
    if model_loaded:
        device = str(next(model_manager.model.parameters()).device)
    
    return jsonify({
        'status': 'healthy',
        'ocr_engine': 'OlmOCR-7B',
        'model': 'allenai/olmOCR-7B-0825-FP8 (4-bit quantized)',
        'quantization': '4-bit',
        'model_loaded': model_loaded,
        'cuda_available': torch.cuda.is_available(),
        'device': device
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
# MODEL SELECTION ROUTES
# ==================

@main_bp.route('/available_models', methods=['GET'])
def get_available_models():
    """Get list of available OCR models for this hardware"""
    return jsonify({
        'success': True,
        'models': model_manager.get_available_models(),
        'selected': model_manager.get_selected_model(),
        'needs_selection': model_manager.needs_model_selection,
        'cuda_available': torch.cuda.is_available()
    })


@main_bp.route('/select_model', methods=['POST'])
def select_model():
    """Select and download an OCR model"""
    try:
        data = request.get_json()
        
        if not data or 'model_id' not in data:
            return jsonify({'error': 'model_id is required'}), 400
        
        model_id = data['model_id']
        
        # Validate and save selection
        model_manager.set_selected_model(model_id)
        model_manager.needs_model_selection = False
        
        # Trigger model download in background (will be handled by initialize_models)
        import threading
        def download_async():
            try:
                model_manager.check_and_download_models()
            except Exception as e:
                logger.error(f"Error downloading model: {e}")
        
        threading.Thread(target=download_async, daemon=True).start()
        
        return jsonify({
            'success': True,
            'message': f'Model {model_id} selected, download started',
            'model_id': model_id
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error selecting model: {e}")
        return jsonify({'error': str(e)}), 500


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
        # Lazy load model only when needed
        model, processor = model_manager.ensure_olmocr_loaded()
        
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
            )
        
        # Decode generated tokens
        input_len = inputs.input_ids.shape[1]
        generated_ids = [output_id[input_len:] for output_id in output_ids]
        generated_text = processor.batch_decode(
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
        
        # Note: Model will auto-unload after 60s of inactivity (managed by timer)
        
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
        
        # Note: Model will auto-unload after 60s of inactivity (managed by timer)
        
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
        
        # Load Qwen model (lazy loading - only on first use)
        logger.info(f"📦 Loading/checking Qwen model...")
        tokenizer, qwen_model = model_manager.ensure_qwen_loaded()
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
        
        # Note: Model will auto-unload after 60s of inactivity (managed by timer)
        
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


# ==================
# PROJECT ROUTES
# ==================

@project_bp.route('/create', methods=['POST'])
def create_project():
    """Create a new project"""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data:
            return jsonify({'error': 'Project name is required'}), 400
        
        project_name = data['name']
        description = data.get('description', '')
        
        metadata = project_manager.create_project(project_name, description)
        
        logger.info(f"✅ Created project: {metadata['project_id']}")
        
        return jsonify({
            'success': True,
            'project': metadata
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error creating project: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/list', methods=['GET'])
def list_projects():
    """List all projects"""
    try:
        projects = project_manager.list_projects()
        
        return jsonify({
            'success': True,
            'projects': projects
        })
        
    except Exception as e:
        logger.error(f"❌ Error listing projects: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get project details"""
    try:
        project = project_manager.get_project(project_id)
        
        if not project:
            return jsonify({'error': 'Project not found'}), 404
        
        return jsonify({
            'success': True,
            'project': project
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting project: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project"""
    try:
        success = project_manager.delete_project(project_id)
        
        if not success:
            return jsonify({'error': 'Project not found'}), 404
        
        logger.info(f"🗑️ Deleted project: {project_id}")
        
        return jsonify({
            'success': True,
            'message': 'Project deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Error deleting project: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/upload_images', methods=['POST'])
def upload_images(project_id):
    """Upload images to project"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        # Get project path
        images_path = project_manager.get_project_path(project_id, 'original_images')
        
        if not images_path:
            return jsonify({'error': 'Project not found'}), 404
        
        uploaded_count = 0
        
        for file in files:
            if file.filename:
                # Save file
                file_path = images_path / file.filename
                file.save(str(file_path))
                uploaded_count += 1
        
        # Update project metadata
        total_images = project_manager.count_files(project_id, 'original_images')
        project_manager.update_workflow_status(project_id, {
            'images_loaded': True,
            'images_count': total_images
        })
        
        logger.info(f"📁 Uploaded {uploaded_count} images to project {project_id}")
        
        return jsonify({
            'success': True,
            'uploaded': uploaded_count,
            'total': total_images
        })
        
    except Exception as e:
        logger.error(f"❌ Error uploading images: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/images', methods=['GET'])
def get_project_images(project_id):
    """Get list of images in project"""
    try:
        folder_type = request.args.get('folder', 'original_images')
        images = project_manager.get_images_list(project_id, folder_type)
        
        return jsonify({
            'success': True,
            'images': images,
            'count': len(images)
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting images: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/image/<path:image_name>', methods=['GET'])
def get_project_image(project_id, image_name):
    """Get a specific image from project (with persistent thumbnail caching)"""
    try:
        folder_type = request.args.get('folder', 'original_images')
        thumbnail = request.args.get('thumbnail', 'false').lower() == 'true'
        
        image_path = project_manager.get_project_path(project_id, folder_type) / image_name
        
        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        # If thumbnail requested, check cache first
        if thumbnail:
            # Check if cached thumbnail exists
            thumbnail_path = project_manager.get_project_path(project_id, 'thumbnails') / image_name
            
            if thumbnail_path.exists():
                # Return cached thumbnail
                return send_file(str(thumbnail_path), mimetype='image/jpeg')
            else:
                # Generate and cache thumbnail
                img = Image.open(str(image_path))
                
                # Create thumbnail (max 200px on longest side)
                max_size = 200
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Ensure thumbnails folder exists
                thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save thumbnail to cache
                img.save(str(thumbnail_path), 'JPEG', quality=85, optimize=True)
                
                return send_file(str(thumbnail_path), mimetype='image/jpeg')
        else:
            return send_file(str(image_path), mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"❌ Error getting image: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/annotations/<image_name>', methods=['GET'])
def get_annotations(project_id, image_name):
    """Get annotations for an image"""
    try:
        annotations = project_manager.load_annotation_data(project_id, image_name)
        
        if annotations is None:
            return jsonify({
                'success': True,
                'annotations': None
            })
        
        return jsonify({
            'success': True,
            'annotations': annotations
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting annotations: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/annotations/<image_name>', methods=['POST'])
def save_annotations(project_id, image_name):
    """Save annotations for an image"""
    try:
        data = request.get_json()
        
        if not data or 'annotations' not in data:
            return jsonify({'error': 'Annotations data required'}), 400
        
        success = project_manager.save_annotation_data(
            project_id, 
            image_name, 
            data['annotations']
        )
        
        if not success:
            return jsonify({'error': 'Failed to save annotations'}), 500
        
        logger.info(f"💾 Saved annotations for {image_name} in project {project_id}")
        
        return jsonify({
            'success': True,
            'message': 'Annotations saved successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Error saving annotations: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/save_cropped', methods=['POST'])
def save_cropped_drawing(project_id):
    """Save a cropped drawing image"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data or 'filename' not in data:
            return jsonify({'error': 'Image data and filename required'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get folder type (default: cropped_drawings, can also be cleaned_drawings)
        folder_type = data.get('folder', 'cropped_drawings')
        
        # Save to specified folder
        target_path = project_manager.get_project_path(project_id, folder_type)
        
        if not target_path:
            return jsonify({'error': 'Project not found'}), 404
        
        filename = data['filename']
        image_path = target_path / filename
        image.save(str(image_path))
        
        logger.info(f"✂️ Saved image to {folder_type}: {filename}")
        
        return jsonify({
            'success': True,
            'message': f'Image saved to {folder_type} successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Error saving image: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/save_ocr_results', methods=['POST'])
def save_ocr_results(project_id):
    """Save OCR results to project"""
    try:
        data = request.get_json()
        
        if not data or 'results' not in data:
            return jsonify({'error': 'OCR results required'}), 400
        
        success = project_manager.save_ocr_results(project_id, data['results'])
        
        if not success:
            return jsonify({'error': 'Failed to save OCR results'}), 500
        
        # Update workflow status
        project_manager.update_workflow_status(project_id, {
            'ocr_processed': len(data['results'])
        })
        
        logger.info(f"💾 Saved OCR results for project {project_id}")
        
        return jsonify({
            'success': True,
            'message': 'OCR results saved successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Error saving OCR results: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/ocr_results', methods=['GET'])
def get_ocr_results(project_id):
    """Get latest OCR results from project"""
    try:
        results = project_manager.get_latest_ocr_results(project_id)
        
        if results is None:
            return jsonify({
                'success': True,
                'results': []
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting OCR results: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/save_ocr_corrections', methods=['POST'])
def save_ocr_corrections(project_id):
    """Save OCR corrections to project"""
    try:
        data = request.get_json()
        
        if not data or 'corrections' not in data:
            return jsonify({'error': 'OCR corrections required'}), 400
        
        success = project_manager.save_ocr_corrections(project_id, data['corrections'])
        
        if not success:
            return jsonify({'error': 'Failed to save OCR corrections'}), 500
        
        logger.info(f"💾 Saved OCR corrections for project {project_id}")
        
        return jsonify({
            'success': True,
            'message': 'OCR corrections saved successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Error saving OCR corrections: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/ocr_corrections', methods=['GET'])
def get_ocr_corrections(project_id):
    """Get OCR corrections from project"""
    try:
        corrections = project_manager.get_ocr_corrections(project_id)
        
        if corrections is None:
            return jsonify({
                'success': True,
                'corrections': {}
            })
        
        return jsonify({
            'success': True,
            'corrections': corrections
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting OCR corrections: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/workflow_status', methods=['POST'])
def update_workflow_status(project_id):
    """Update workflow status"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Status updates required'}), 400
        
        success = project_manager.update_workflow_status(project_id, data)
        
        if not success:
            return jsonify({'error': 'Project not found'}), 404
        
        return jsonify({
            'success': True,
            'message': 'Workflow status updated'
        })
        
    except Exception as e:
        logger.error(f"❌ Error updating workflow status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/file/<path:filename>', methods=['GET'])
def get_project_file(project_id, filename):
    """Get a file from project (JSON, text, etc.)"""
    try:
        folder = request.args.get('folder', 'ocr_results')
        project_path = project_manager.get_project_path(project_id, folder)
        
        if not project_path:
            return jsonify({'error': 'Project not found'}), 404
        
        file_path = project_path / filename
        
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        # Read and return JSON files as JSON
        if filename.endswith('.json'):
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data)
        
        # For other files, send as attachment
        return send_file(file_path)
        
    except Exception as e:
        logger.error(f"❌ Error getting file: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/save_fewshot_examples', methods=['POST'])
def save_fewshot_examples(project_id):
    """Save few-shot examples to project"""
    try:
        data = request.get_json()
        
        if not data or 'examples' not in data:
            return jsonify({'error': 'Few-shot examples required'}), 400
        
        examples = data['examples']
        
        success = project_manager.save_fewshot_examples(project_id, examples)
        
        if not success:
            return jsonify({'error': 'Failed to save few-shot examples'}), 500
        
        logger.info(f"✅ Saved {len(examples)} few-shot examples to project {project_id}")
        
        return jsonify({
            'success': True,
            'message': f'Saved {len(examples)} few-shot examples',
            'count': len(examples)
        })
        
    except Exception as e:
        logger.error(f"❌ Error saving few-shot examples: {str(e)}")
        return jsonify({'error': str(e)}), 500


@project_bp.route('/<project_id>/fewshot_examples', methods=['GET'])
def get_fewshot_examples(project_id):
    """Get few-shot examples from project"""
    try:
        examples = project_manager.get_fewshot_examples(project_id)
        
        if examples is None:
            return jsonify({
                'success': True,
                'examples': [],
                'count': 0,
                'message': 'No few-shot examples found'
            })
        
        logger.info(f"📚 Loaded {len(examples)} few-shot examples from project {project_id}")
        
        return jsonify({
            'success': True,
            'examples': examples,
            'count': len(examples)
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting few-shot examples: {str(e)}")
        return jsonify({'error': str(e)}), 500
