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
    app.jinja_env.auto_reload = True
    
    # Override with custom config if provided
    if config:
        app.config.update(config)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    from app.routes import bp, start_auto_shutdown_watchdog
    app.register_blueprint(bp)

    # Auto-shutdown: browser tab <-> this process heartbeat/beacon (see AUTO_SHUTDOWN_MODULES_GUIDE.md)
    start_auto_shutdown_watchdog()

    logger.info("Flask app created successfully")
    
    return app
