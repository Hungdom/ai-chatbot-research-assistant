from src.database.session import init_db
from src.crawler.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    try:
        logger.info("Initializing database...")
        engine = init_db()
        logger.info("Database initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    main() 