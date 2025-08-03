import traceback
import psycopg2
import sys
from core.config import Config
from core.secure_config import SecureConfig, ConfigurationError


def get_neon_connection_string():
    """Get Neon connection string from config with fallback to secure config."""
    # First try the main config system
    if Config.database.is_configured():
        return Config.database.NEON_CONN_STR
    
    # Fallback to secure config for backward compatibility
    try:
        secure_config = SecureConfig()
        return secure_config.NEON_CONN_STR
    except ConfigurationError as e:
        print(f"SECURITY ERROR: {e}")
        return None


def log_error_to_neon(user_name: str, exc_type: type, exc_value: Exception, tb) -> None:
    """Log error to Neon database with proper connection management."""
    # Get connection string from config
    neon_conn_str = get_neon_connection_string()
    if not neon_conn_str:
        print("NEON_CONN_STR environment variable is not set or database not configured.")
        return

    try:
        #Extract file name and line number from traceback (support the debugging)
        extracted_tb = traceback.extract_tb(tb)
        if extracted_tb:
            last_frame = extracted_tb[-1]
            file_name = last_frame.filename
            line_number = last_frame.lineno
        else:
            file_name = "Unknown"
            line_number = -1

        error_message = str(exc_value)[:500]    # To prevent long message crashes
        error_type = exc_type.__name__[:50]
        user_name = str(user_name)[:100]
        file_name = file_name[:200]

        with psycopg2.connect(neon_conn_str, connect_timeout=Config.database.CONNECTION_TIMEOUT) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO error_logs_custom_predictions 
                    (user_name, error_message, error_type, file_name, line_number)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (user_name, error_message, error_type, file_name, line_number)
                )
                conn.commit()

    except psycopg2.Error as db_exc:
        print(f"Failed to log error to the Database: {db_exc}")

    except Exception as e:
        print(f"Unexpected logging error: {e}")


def global_exception_hook(exctype, value, tb):
    try:
        #retrieve user_name from a window object if it exists
        window = getattr(sys.modules['__main__'], "window", None)
        user_name = getattr(window, "user_name", "UnknownUser") if window else "UnknownUser"
    except Exception:
        user_name = "UnknownUser"  # fallback to default name if any error

    #Loggin the exception details using the secure function
    log_error_to_neon(user_name, exctype, value, tb)

    #Calling the original system exception handler so default behaviour continues
    sys.__excepthook__(exctype, value, tb)
