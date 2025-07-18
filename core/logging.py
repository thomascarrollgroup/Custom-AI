import traceback
import psycopg2
import sys
import os

NEON_CONN_STR = os.getenv("NEON_CONN_STR")

def log_error_to_neon(user_name, exc_type, exc_value, tb):
    try:
        tb_last = traceback.extract_tb(tb)[-1]
        file_name = tb_last.filename
        line_number = tb_last.lineno
        error_message = str(exc_value)
        error_type = exc_type.__name__

        conn = psycopg2.connect(NEON_CONN_STR)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO error_logs_custom_predictions (user_name, error_message, error_type, file_name, line_number)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_name, error_message, error_type, file_name, line_number)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as log_exc:
        print("Failed to log error to Neon:", log_exc)

def global_exception_hook(exctype, value, tb):
    user_name = None
    try:
        window = getattr(sys.modules['__main__'], "window", None)
        if window and hasattr(window, "user_name"):
            user_name = window.user_name
        else:
            user_name = "UnknownUser"
    except Exception:
        user_name = "UnknownUser"
    log_error_to_neon(user_name, exctype, value, tb)
    sys.__excepthook__(exctype, value, tb)