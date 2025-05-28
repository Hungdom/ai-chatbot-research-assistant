from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import openai
from sqlalchemy import create_engine, Column, Integer, String, Text, ARRAY, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Research Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres_db:5432/research_db")
logger.info(f"Connecting to database: {DATABASE_URL}")

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    # Models
    class Paper(Base):
        __tablename__ = "papers"
        
        id = Column(Integer, primary_key=True, index=True)
        title = Column(String, index=True)
        abstract = Column(Text)
        authors = Column(ARRAY(String))
        keywords = Column(ARRAY(String))
        year = Column(Integer)

    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Database connection error: {str(e)}")
    raise

# Pydantic models for request/response
class ChatRequest(BaseModel):
    query: str
    year_filter: Optional[int] = None
    keywords: Optional[List[str]] = None

class ChatResponse(BaseModel):
    response: str
    papers: List[dict]

# Initialize OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai.api_key = openai_api_key
logger.info("OpenAI API configured successfully")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.query}")
        
        # Get database session
        db = SessionLocal()
        
        try:
            # Query papers (basic implementation - can be enhanced with semantic search)
            papers_query = db.query(Paper)
            
            # Apply year filter if provided
            if request.year_filter:
                papers_query = papers_query.filter(Paper.year == request.year_filter)
            
            # Apply keywords filter if provided
            if request.keywords:
                # Use PostgreSQL's && operator for array overlap
                papers_query = papers_query.filter(
                    Paper.keywords.op('&&')(request.keywords)
                )
            
            papers = papers_query.all()
            logger.info(f"Found {len(papers)} relevant papers")
            
            # If no papers found, return a default response
            if not papers:
                return ChatResponse(
                    response="I couldn't find any papers matching your criteria. Try adjusting your filters or asking a different question.",
                    papers=[]
                )
            
            # Prepare context for GPT
            context = "\n".join([
                f"Title: {paper.title}\nAbstract: {paper.abstract}\nAuthors: {', '.join(paper.authors)}\n"
                for paper in papers[:5]  # Limit to top 5 papers for context
            ])
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant. Use the provided paper information to answer the user's query."},
                    {"role": "user", "content": f"Query: {request.query}\n\nRelevant papers:\n{context}"}
                ]
            )
            
            return ChatResponse(
                response=response.choices[0].message.content,
                papers=[{
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "year": paper.year,
                    "keywords": paper.keywords
                } for paper in papers]
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Research Assistant API is running"} 