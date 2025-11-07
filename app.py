#!/usr/bin/env python3
"""
PyPotteryScan - Archaeological Drawing Processor
Main entry point for running the Flask application
"""
import os
import sys
import webbrowser
import threading
import time
import logging

# Add app to path
sys.path.insert(0, os.path.dirname(__file__))

from app import create_app
from app.model_manager import model_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def open_browser(port):
    """Open browser after brief delay"""
    time.sleep(1.5)
    url = f"http://localhost:{port}/"
    logger.info(f"🌐 Opening browser: {url}")
    webbrowser.open(url)


def load_models_async(app):
    """Load models in background thread"""
    try:
        with app.app_context():
            logger.info("📦 Starting model initialization...")
            model_manager.initialize_models()
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")


if __name__ == '__main__':
    try:
        # Create Flask app
        app = create_app()
        
        port = app.config.get('PORT', 5002)
        host = app.config.get('HOST', '0.0.0.0')
        
        logger.info("=" * 60)
        logger.info("🏺 PyPotteryScan - Archaeological Drawing Processor")
        logger.info("=" * 60)
        logger.info(f"🚀 Starting server on http://localhost:{port}")
        logger.info(f"📍 Health check: http://localhost:{port}/health")
        logger.info(f"📍 Loading status: http://localhost:{port}/loading_status")
        logger.info("=" * 60)
        
        # Initialize model_manager with app config
        from app.config import Config
        model_manager.config = Config.get_config_dict()
        
        # Open browser immediately
        threading.Thread(target=open_browser, args=(port,), daemon=True).start()
        
        # Load models in background
        threading.Thread(target=load_models_async, args=(app,), daemon=True).start()
        
        # Start Flask server with threading enabled
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        sys.exit(1)
