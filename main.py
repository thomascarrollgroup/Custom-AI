# CRITICAL FIX: Prevent infinite window spawning
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent window spawning

# Prevent multiple QApplication instances
os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Import matplotlib after setting backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

import sys
import os
from PyQt5.QtWidgets import QApplication
from ui.app import BespokePredictionApp
from core.error_logger import log_error_to_file
from core.logging_setup import setup_logging, get_logger

if __name__ == "__main__":
    # Initialize comprehensive logging system
    metrics_collector = setup_logging(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        enable_console=True,
        enable_file=True,
        enable_json=True
    )
    
    # Set up global exception handling with file-based logging
    def global_exception_hook(exctype, value, tb):
        try:
            # Try to get user name from the main window if it exists
            window = getattr(sys.modules['__main__'], "window", None)
            user_name = getattr(window, "user_name", "UnknownUser") if window else "UnknownUser"
        except Exception:
            user_name = "UnknownUser"
        
        # Log error to file
        log_error_to_file(user_name, exctype, value, tb)
        
        # Call the original system exception handler
        sys.__excepthook__(exctype, value, tb)
    
    sys.excepthook = global_exception_hook
    
    # Get logger for main application
    logger = get_logger(__name__)
    logger.info("Starting TC AI Prediction Tool")
    logger.info("Application Error logging system configured successfully")
    
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