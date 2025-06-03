from sqlalchemy import create_engine, Column, Integer, String, Text, text, DateTime, JSON, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from pgvector.sqlalchemy import Vector
import os
from contextlib import contextmanager
from datetime import datetime

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
    
    id = Column(Integer, primary_key=True)
    arxiv_id = Column(String(50), unique=True, index=True)
    title = Column(String(500), index=True)
    abstract = Column(Text)
    authors = Column(ARRAY(String))
    categories = Column(ARRAY(String))
    published_date = Column(DateTime, index=True)
    updated_date = Column(DateTime)
    doi = Column(String(100), index=True)
    journal_ref = Column(String(500))
    primary_category = Column(String(50), index=True)
    comment = Column(Text)
    embedding = Column(Vector(1536))  # OpenAI embeddings are 1536-dimensional
    
    # Add composite indexes for common query patterns
    __table_args__ = (
        Index('idx_published_categories', published_date, categories),
        Index('idx_published_authors', published_date, authors),
        Index('idx_title_abstract', title, abstract),
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