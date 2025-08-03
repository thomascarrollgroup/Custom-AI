import sys
import os
from PyQt5.QtWidgets import QApplication
from ui.app import BespokePredictionApp
from core.logging import global_exception_hook
from core.logging_setup import setup_logging, get_logger

if __name__ == "__main__":
    # Initialize comprehensive logging system
    metrics_collector = setup_logging(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        enable_console=True,
        enable_file=True,
        enable_json=True
    )
    
    # Set up global exception handling
    sys.excepthook = global_exception_hook
    
    # Get logger for main application
    logger = get_logger(__name__)
    logger.info("Starting TC AI Prediction Tool")
    
    # Initialize and validate database connection
    try:
        from core.database_utils import initialize_database, get_database_status
        
        logger.info("Validating database configuration...")
        db_status = get_database_status()
        
        if db_status['connection_string_set']:
            if initialize_database():
                logger.info("Neon database initialized successfully- error logging enabled")
            else:
                logger.warning("Neon database initialization failed- error logging to console only")
        else:
            logger.warning("NEON_CONN_STR not set- error logging to console only")
            logger.info("Set NEON_CONN_STR environment variable to enable database error logging")
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        logger.warning("Continuing without database error logging")
    
    try:
        app = QApplication(sys.argv)
        window = BespokePredictionApp()
        window.show()
        
        logger.info("Application window displayed")
        exit_code = app.exec_()
        logger.info(f"Application exiting with code: {exit_code}")
        
        # Log final metrics summary
        metrics_summary = metrics_collector.get_metrics_summary()
        if metrics_summary:
            logger.info(f"Session metrics: {metrics_summary}")
        
        sys.exit(exit_code)
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)