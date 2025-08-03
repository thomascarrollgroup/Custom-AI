import psycopg2
from typing import Optional
from core.config import Config
from core.logging import get_neon_connection_string
from core.logging_setup import get_logger


def validate_neon_connection() -> bool:
    """
    Validate that Neon database connection is properly configured and accessible.
    
    Returns:
        bool: True if connection is valid, False otherwise
    """
    logger = get_logger(__name__)
    
    try:
        neon_conn_str = get_neon_connection_string()
        if not neon_conn_str:
            logger.warning("Neon connection string not configured")
            return False
        
        # Test the connection
        with psycopg2.connect(neon_conn_str, connect_timeout=Config.database.CONNECTION_TIMEOUT) as conn:
            with conn.cursor() as cur:
                # Test basic connectivity
                cur.execute("SELECT 1")
                result = cur.fetchone()
                
                if result and result[0] == 1:
                    logger.info("Neon database connection validated successfully")
                    return True
                else:
                    logger.error("Neon database connection test failed")
                    return False
                    
    except psycopg2.OperationalError as e:
        logger.error(f"Neon database connection failed: {e}")
        return False
    except psycopg2.Error as e:
        logger.error(f"Neon database error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error validating Neon connection: {e}")
        return False


def ensure_error_table_exists() -> bool:
    """
    Ensure the error logging table exists in Neon database.
    
    Returns:
        bool: True if table exists or was created successfully, False otherwise
    """
    logger = get_logger(__name__)
    
    try:
        neon_conn_str = get_neon_connection_string()
        if not neon_conn_str:
            logger.warning("Neon connection string not configured")
            return False
        
        with psycopg2.connect(neon_conn_str, connect_timeout=Config.database.CONNECTION_TIMEOUT) as conn:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'error_logs_custom_predictions'
                    );
                """)
                
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    logger.info("Creating error_logs_custom_predictions table")
                    # Create the table if it doesn't exist
                    cur.execute("""
                        CREATE TABLE error_logs_custom_predictions (
                            id SERIAL PRIMARY KEY,
                            user_name VARCHAR(100),
                            error_message TEXT,
                            error_type VARCHAR(50),
                            file_name VARCHAR(200),
                            line_number INTEGER,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    conn.commit()
                    logger.info("Error logging table created successfully")
                else:
                    logger.info("Error logging table already exists")
                
                return True
                
    except psycopg2.Error as e:
        logger.error(f"Failed to ensure error table exists: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error ensuring error table: {e}")
        return False


def get_database_status() -> dict:
    """
    Get comprehensive database status information.
    
    Returns:
        dict: Database status information
    """
    status = {
        'configured': False,
        'connection_valid': False,
        'error_table_exists': False,
        'connection_string_set': False
    }
    
    # Check if connection string is configured
    neon_conn_str = get_neon_connection_string()
    status['connection_string_set'] = bool(neon_conn_str)
    status['configured'] = Config.database.is_configured()
    
    if status['configured']:
        status['connection_valid'] = validate_neon_connection()
        if status['connection_valid']:
            status['error_table_exists'] = ensure_error_table_exists()
    
    return status


def initialize_database() -> bool:
    """
    Initialize database connection and ensure all required tables exist.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    logger = get_logger(__name__)
    logger.info("Initializing database connection")
    
    status = get_database_status()
    
    if not status['configured']:
        logger.warning("Database not configured - error logging to Neon will be disabled")
        return False
    
    if not status['connection_valid']:
        logger.error("Database connection invalid - error logging to Neon will be disabled")
        return False
    
    if not status['error_table_exists']:
        logger.error("Error logging table could not be created - error logging to Neon will be disabled")
        return False
    
    logger.info("Database initialization completed successfully")
    return True
