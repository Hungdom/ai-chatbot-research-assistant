from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv
import logging
import uvicorn
from database import get_db, Base, engine, ChatSession, Arxiv
from agents.chat_agent import ChatAgent
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
import json
from contextlib import contextmanager
from sqlalchemy.sql import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set uvicorn access log format
log_config = uvicorn.config.LOGGING_CONFIG
log_config["formatters"]["access"]["fmt"] = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_config["formatters"]["default"]["fmt"] = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Research Assistant Chatbot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
try:
    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Database connection error: {str(e)}")
    raise

# Initialize chat agent
chat_agent = ChatAgent()

# Request/Response Models
class ChatMessage(BaseModel):
    role: str
    content: str
    html_content: Optional[str] = None  # Add HTML content field
    timestamp: str

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    html_response: Optional[str] = None
    arxiv: Optional[List[Dict[str, Any]]] = None
    intent: Optional[Dict[str, Any]] = None
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    context: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

# Add SearchRequest model
class SearchRequest(BaseModel):
    year: Optional[int] = None
    keywords: Optional[List[str]] = []

# Add SearchResponse model
class SearchResponse(BaseModel):
    papers: List[Dict[str, Any]]
    summary: str

# API Endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and return a response.
    """
    try:
        # Get or create session
        with get_db() as db:
            if request.session_id:
                session = db.query(ChatSession).filter(ChatSession.session_id == request.session_id).first()
            else:
                session = None

            if not session:
                session = ChatSession(
                    session_id=str(uuid.uuid4()),
                    messages=[],
                    context={}
                )
                db.add(session)
                db.commit()
                logger.info(f"Created new session: {session.session_id}")

            # Process the chat request
            response_data = await chat_agent.process({
                "query": request.query,
                "session_id": session.session_id,
                "context": session.context
            })

            # Create response message with timestamp
            current_time = datetime.utcnow().isoformat()
            response_message = ChatMessage(
                role="assistant",
                content=response_data.get("response", ""),
                html_content=response_data.get("html_response", response_data.get("response", "")),
                timestamp=current_time
            )

            # Update session messages
            if isinstance(session.messages, str):
                messages = json.loads(session.messages)
            else:
                messages = session.messages or []

            # Add user message with timestamp
            user_message = {
                "role": "user",
                "content": request.query,
                "timestamp": current_time
            }
            messages.append(user_message)

            # Add assistant response
            messages.append(response_message.dict())

            # Update session
            session.messages = messages
            session.context = {
                **(session.context or {}),
                "arxiv": response_data.get("arxiv", []),
                "intent": response_data.get("intent", {})
            }
            session.updated_at = datetime.utcnow()
            db.commit()
            logger.info(f"Updated session {session.session_id} with new messages")

            return ChatResponse(
                response=response_data.get("response", ""),
                html_response=response_data.get("html_response", ""),
                arxiv=response_data.get("arxiv", []),
                intent=response_data.get("intent", {}),
                session_id=session.session_id
            )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions", response_model=List[SessionResponse])
async def get_sessions():
    """List all chat sessions."""
    try:
        with get_db() as db:
            sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
            logger.info(f"Retrieved {len(sessions)} sessions")
            
            # Process sessions to ensure all messages have timestamps
            processed_sessions = []
            for session in sessions:
                messages = session.messages or []
                # Ensure all messages have timestamps
                processed_messages = []
                for msg in messages:
                    if isinstance(msg, dict):
                        if 'timestamp' not in msg:
                            msg['timestamp'] = datetime.utcnow().isoformat()
                        processed_messages.append(msg)
                
                processed_sessions.append({
                    "session_id": session.session_id,
                    "messages": processed_messages,
                    "context": session.context or {},
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                    "updated_at": session.updated_at.isoformat() if session.updated_at else None
                })
            
            return processed_sessions
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get a chat session by ID."""
    try:
        with get_db() as db:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Convert messages to list if they're stored as JSON string
            if isinstance(session.messages, str):
                messages = json.loads(session.messages)
            else:
                messages = session.messages or []
            
            # Ensure all messages have timestamps
            processed_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    if 'timestamp' not in msg:
                        msg['timestamp'] = datetime.utcnow().isoformat()
                    processed_messages.append(msg)
            
            return {
                "session_id": session.session_id,
                "messages": processed_messages,
                "context": session.context or {},
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None
            }
    except Exception as e:
        logger.error(f"Error retrieving session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/sessions/{session_id}", response_model=Dict[str, str])
async def delete_session(session_id: str):
    """Delete a chat session."""
    try:
        with get_db() as db:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if session:
                db.delete(session)
                db.commit()
                return {"message": "Session deleted successfully"}
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/sessions/{session_id}")
async def update_session(session_id: str, session_data: dict):
    try:
        with get_db() as db:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Update session data
            if "messages" in session_data:
                session.messages = session_data["messages"]
            if "context" in session_data:
                session.context = session_data["context"]
            
            # Update timestamp
            session.updated_at = datetime.utcnow()
            
            # Commit changes
            db.commit()
            logger.info(f"Updated session {session_id} with new messages")
            
            return {"status": "success", "message": "Session updated successfully"}
    except Exception as e:
        logger.error(f"Error updating session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Check if the API is healthy."""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Research Assistant Chatbot API is running"}

# Add search endpoint
@app.post("/api/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """
    Search for papers in the arXiv database using both semantic and keyword-based search.
    """
    try:
        with get_db() as db:
            # Get query embedding for semantic search
            query_embedding = await chat_agent._get_embedding(
                " ".join(request.keywords) if request.keywords else ""
            )
            
            # Build the base query
            query = db.query(Arxiv)
            
            # Add year filter if provided
            if request.year:
                query = query.filter(Arxiv.published_date >= f"{request.year}-01-01",
                                   Arxiv.published_date < f"{request.year + 1}-01-01")
            
            # Add keyword filter if provided
            if request.keywords and len(request.keywords) > 0:
                keyword_conditions = []
                for keyword in request.keywords:
                    keyword_lower = keyword.lower()
                    keyword_conditions.append(
                        (Arxiv.title.ilike(f"%{keyword_lower}%") | 
                         Arxiv.abstract.ilike(f"%{keyword_lower}%"))
                    )
                query = query.filter(*keyword_conditions)
            
            # Add semantic similarity if we have embeddings
            if query_embedding:
                # Convert embedding to string format for PostgreSQL
                embedding_str = str(query_embedding).replace("'", "")
                # Add cosine similarity calculation
                similarity_expr = text(
                    "cosine_similarity(arxiv.embedding::vector, ARRAY" + embedding_str + "::vector) as similarity"
                )
                query = query.add_columns(similarity_expr)
                # Order by similarity if available
                query = query.order_by(text("similarity DESC"))
            else:
                # Fallback to date ordering if no embeddings
                query = query.order_by(Arxiv.published_date.desc())
            
            # Execute query with limit
            results = query.limit(50).all()
            
            # Convert papers to dict format
            papers_data = []
            for result in results:
                if isinstance(result, tuple):
                    paper, similarity = result
                else:
                    paper = result
                    similarity = None
                
                paper_dict = {
                    "id": paper.id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract,
                    "categories": paper.categories,
                    "published_date": paper.published_date.isoformat() if paper.published_date else None,
                    "updated_date": paper.updated_date.isoformat() if paper.updated_date else None,
                    "doi": paper.doi,
                    "journal_ref": paper.journal_ref,
                    "primary_category": paper.primary_category,
                    "comment": paper.comment,
                    "similarity": float(similarity) if similarity is not None else None
                }
                papers_data.append(paper_dict)
            
            # Generate enhanced summary
            summary = ""
            if papers_data:
                year_text = f"in {request.year}" if request.year else ""
                keyword_text = f"related to {', '.join(request.keywords)}" if request.keywords else ""
                similarity_text = "using semantic similarity" if query_embedding else ""
                summary = f"Found {len(papers_data)} papers {year_text} {keyword_text} {similarity_text}."
            
            return SearchResponse(
                papers=papers_data,
                summary=summary
            )
            
    except Exception as e:
        logger.error(f"Error searching papers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True) 