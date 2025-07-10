from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, DateTime, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from pgvector.sqlalchemy import Vector
import os
from contextlib import contextmanager
from datetime import datetime
from sqlalchemy.types import Float

# Database connection parameters
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),  # Update default to localhost
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres_kltn')
}

# Database URL
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Create engine with connection pool settings
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections after 30 minutes
    echo=True  # Enable SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Define ArXiv model
class Arxiv(Base):
    __tablename__ = "arxiv"
    
    id = Column(Integer, primary_key=True, index=True)
    arxiv_id = Column(String(50), unique=True, index=True)
    title = Column(Text, index=True)  # Add index for title searches
    abstract = Column(Text)
    authors = Column(ARRAY(String))
    categories = Column(ARRAY(String), index=True)  # Add index for category searches
    published_date = Column(DateTime, index=True)  # Add index for date sorting
    doi = Column(String(100))
    primary_category = Column(String(50), index=True)  # Add index for primary category
    embedding = Column(ARRAY(Float), nullable=True)

    # Add composite indexes for common query patterns
    __table_args__ = (
        Index('idx_published_date_desc', published_date.desc()),  # For recent papers
        Index('idx_primary_category_date', primary_category, published_date.desc()),  # For category + date
        Index('idx_title_search', text('title gin_trgm_ops'), postgresql_using='gin'),  # For title text search
        Index('idx_categories_gin', categories, postgresql_using='gin'),  # For category array searches
    )

# Define ChatSession model
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), unique=True, index=True)
    messages = Column(JSONB, default=list)
    context = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Add indexes for common query patterns
    __table_args__ = (
        Index('idx_session_updated', updated_at),
        Index('idx_session_created', created_at),
    )

# Context manager for database sessions
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database tables
def init_db():
    try:
        # Create vector extension if it doesn't exist
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Create function to cast array to vector
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION array_to_vector(arr numeric[])
                RETURNS vector
                AS $$
                BEGIN
                    RETURN arr::vector;
                END;
                $$ LANGUAGE plpgsql IMMUTABLE;
            """))
            
            # Create cosine similarity function
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
                RETURNS float
                AS $$
                BEGIN
                    RETURN 1 - (a <=> b);
                END;
                $$ LANGUAGE plpgsql IMMUTABLE;
            """))
            
            conn.commit()
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating database tables: {str(e)}")
        raise 