import os
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError
from database.models import Base
from config.settings import DB_CONFIG
from crawler.utils.logger import setup_logger

logger = setup_logger(__name__)

def get_database_url():
    """Get database URL from environment variables"""
    host = os.getenv('POSTGRES_HOST', 'postgres')
    port = os.getenv('POSTGRES_PORT', '5432')
    db = os.getenv('POSTGRES_DB', 'scholar_db')
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD', 'postgres')
    
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

def wait_for_db(max_retries=5, retry_interval=2):
    """Wait for database to be ready"""
    engine = create_engine(get_database_url())
    for attempt in range(max_retries):
        try:
            engine.connect()
            logger.info("Database connection successful")
            return True
        except OperationalError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}. Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
            else:
                logger.error(f"Could not connect to database after {max_retries} attempts")
                raise

def init_db():
    """Initialize database tables"""
    engine = create_engine(get_database_url())
    Base.metadata.create_all(engine)
    logger.info("Database tables initialized")

# Wait for database to be ready
wait_for_db()

# Create a single engine instance with connection pooling
engine = create_engine(
    get_database_url(),
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    echo=True  # Enable SQL logging
)
logger.info("Database engine created with connection pooling")

# Initialize database tables
init_db()

# Create a session factory with scoped sessions
Session = scoped_session(
    sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=True,
        expire_on_commit=False
    )
)
logger.info("Session factory created")

def get_session():
    """Get a new database session"""
    engine = create_engine(get_database_url())
    Session = sessionmaker(bind=engine)
    return Session() 