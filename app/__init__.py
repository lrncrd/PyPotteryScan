"""
PyPotteryScan Flask Application Factory
"""
from flask import Flask
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(config=None):
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Load default config
    app.config.from_object('app.config.Config')
    
    # Override with custom config if provided
    if config:
        app.config.update(config)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    from app.routes import main_bp, ocr_bp, parser_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(ocr_bp)
    app.register_blueprint(parser_bp)
    
    logger.info("Flask app created successfully")
    
    return app
