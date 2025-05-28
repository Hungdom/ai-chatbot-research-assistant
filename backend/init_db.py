from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import Base, Paper
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres_db:5432/research_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# Sample data
sample_papers = [
    Paper(
        title="Machine Learning in Healthcare",
        abstract="A comprehensive review of machine learning applications in healthcare, focusing on diagnosis and treatment prediction.",
        authors=["John Smith", "Jane Doe"],
        keywords=["machine learning", "healthcare", "AI"],
        year=2023
    ),
    Paper(
        title="Natural Language Processing Advances",
        abstract="Recent developments in natural language processing and their impact on human-computer interaction.",
        authors=["Alice Johnson", "Bob Wilson"],
        keywords=["NLP", "AI", "language models"],
        year=2023
    ),
    Paper(
        title="Deep Learning for Image Recognition",
        abstract="State-of-the-art approaches in deep learning for computer vision and image recognition tasks.",
        authors=["Charlie Brown", "Diana Miller"],
        keywords=["deep learning", "computer vision", "AI"],
        year=2022
    )
]

# Insert sample data
db = SessionLocal()
try:
    for paper in sample_papers:
        db.add(paper)
    db.commit()
    print("Sample data inserted successfully")
except Exception as e:
    print(f"Error inserting sample data: {str(e)}")
    db.rollback()
finally:
    db.close() 