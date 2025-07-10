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
from sqlalchemy import or_, select
from agents.enhanced_chat_agent import EnhancedChatAgent
from agents.smart_search_agent import SmartSearchAgent

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

# Initialize enhanced agents
enhanced_chat_agent = EnhancedChatAgent()
smart_search_agent = SmartSearchAgent()

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
    metadata: Optional[Dict[str, Any]] = None

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
                # Check if we have papers with embeddings
                embedding_check_query = text("SELECT COUNT(*) FROM arxiv WHERE embedding IS NOT NULL")
                embedding_count = db.execute(embedding_check_query).scalar()
                
                if embedding_count > 0:
                    # Convert embedding to proper format for PostgreSQL
                    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                    
                    # Add cosine similarity calculation
                    similarity_expr = text(
                        f"cosine_similarity(arxiv.embedding, '{embedding_str}'::vector) as similarity"
                    )
                    query = query.add_columns(similarity_expr)
                    
                    # Only include papers with embeddings in similarity search
                    query = query.filter(Arxiv.embedding.is_not(None))
                    
                    # Order by similarity if available
                    query = query.order_by(text("similarity DESC"))
                else:
                    logger.warning("No papers with embeddings found, using date ordering")
                    query = query.order_by(Arxiv.published_date.desc())
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

# Add enhanced chat endpoint
@app.post("/enhanced-chat")
async def enhanced_chat(request: ChatRequest):
    """
    Enhanced chat endpoint with smart search capabilities
    """
    logger.info(f"Enhanced chat request: {request.query}")
    
    try:
        # Process with enhanced agent
        result = await enhanced_chat_agent.process({
            "query": request.query,
            "context": request.context or {},
            "session_id": request.session_id
        })
        
        logger.info(f"Enhanced chat completed: {len(result['arxiv'])} papers found")
        
        return ChatResponse(
            response=result["response"],
            html_response=result["html_response"],
            arxiv=result["arxiv"],
            intent=result["intent"],
            session_id=result["session_id"],
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {str(e)}", exc_info=True)
        return ChatResponse(
            response="I apologize, but I encountered an error processing your request. Please try again.",
            html_response="<div class='error'>Error processing request</div>",
            arxiv=[],
            intent={"type": "error"},
            session_id=request.session_id,
            metadata={"error": str(e)}
        )

# Add research landscape analysis endpoint
@app.post("/research-landscape")
async def research_landscape(request: dict):
    """
    Analyze research landscape for a given category or topic
    """
    category = request.get("category", "")
    years = request.get("years", None)
    
    logger.info(f"Research landscape analysis for: {category}")
    
    try:
        analysis = await smart_search_agent.analyze_research_landscape(category, years)
        
        return {
            "category": category,
            "analysis": analysis,
            "years": years
        }
        
    except Exception as e:
        logger.error(f"Research landscape error: {str(e)}", exc_info=True)
        return {"error": str(e)}

# Enhanced search endpoint
@app.post("/smart-search")
async def smart_search(request: dict):
    """
    Smart search endpoint with enhanced capabilities
    """
    query = request.get("query", "")
    intent = request.get("intent", {})
    
    logger.info(f"Smart search request: {query}")
    
    try:
        papers = await smart_search_agent.smart_search(query, intent)
        
        return {
            "query": query,
            "papers": papers,
            "count": len(papers),
            "search_metadata": {
                "categories_analyzed": intent.get("categories", []),
                "temporal_scope": intent.get("temporal_scope", "all"),
                "search_strategy": intent.get("search_strategy", "hybrid")
            }
        }
        
    except Exception as e:
        logger.error(f"Smart search error: {str(e)}", exc_info=True)
        return {"error": str(e), "papers": []}

# Add category analysis endpoint
@app.get("/category-analysis/{category}")
async def category_analysis(category: str):
    """
    Get detailed analysis of a specific category
    """
    logger.info(f"Category analysis for: {category}")
    
    try:
        with get_db() as db:
            # Get expanded categories
            expanded_categories = smart_search_agent._expand_categories([category])
            
            # Build query for category analysis
            category_conditions = []
            for cat in expanded_categories:
                category_conditions.append(
                    or_(
                        Arxiv.categories.contains([cat]),
                        Arxiv.primary_category == cat
                    )
                )
            
            if category_conditions:
                query = select(Arxiv).where(or_(*category_conditions))
                papers = db.execute(query.limit(100)).scalars().all()
                
                # Convert to analysis format
                paper_dicts = [smart_search_agent._format_paper_dict(p) for p in papers]
                
                # Generate analysis
                analysis = {
                    "category": category,
                    "expanded_categories": expanded_categories,
                    "total_papers": len(papers),
                    "category_distribution": smart_search_agent._analyze_category_distribution(papers),
                    "temporal_trends": smart_search_agent._analyze_temporal_trends(papers),
                    "collaboration_patterns": smart_search_agent._analyze_collaboration_patterns(papers),
                    "sample_papers": paper_dicts[:10]
                }
                
                return analysis
            else:
                return {"error": "No papers found for category", "category": category}
                
    except Exception as e:
        logger.error(f"Category analysis error: {str(e)}", exc_info=True)
        return {"error": str(e)}

# Update existing chat endpoint to use enhanced agent as fallback
@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with enhanced capabilities
    """
    logger.info(f"Chat request: {request.query}")
    
    try:
        # Try enhanced agent first
        result = await enhanced_chat_agent.process({
            "query": request.query,
            "context": request.context or {},
            "session_id": request.session_id
        })
        
        logger.info(f"Chat completed: {len(result['arxiv'])} papers found")
        
        return ChatResponse(
            response=result["response"],
            html_response=result["html_response"],
            arxiv=result["arxiv"],
            intent=result["intent"],
            session_id=result["session_id"]
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        # Fallback to basic response
        return ChatResponse(
            response="I apologize, but I encountered an error processing your request. Please try again.",
            html_response="<div class='error'>Error processing request</div>",
            arxiv=[],
            intent={"type": "error"},
            session_id=request.session_id
        )

# Add endpoint for category suggestions
@app.get("/category-suggestions")
async def category_suggestions():
    """
    Get category suggestions for search
    """
    return {
        "main_categories": [
            {"code": "astro-ph", "name": "Astrophysics", "description": "Astronomy and astrophysics research"},
            {"code": "cs", "name": "Computer Science", "description": "Computer science and related fields"},
            {"code": "math", "name": "Mathematics", "description": "Mathematics and mathematical physics"},
            {"code": "physics", "name": "Physics", "description": "Physics research across all subfields"},
            {"code": "hep", "name": "High Energy Physics", "description": "Particle physics and high energy physics"},
            {"code": "gr-qc", "name": "General Relativity", "description": "General relativity and quantum cosmology"},
            {"code": "cond-mat", "name": "Condensed Matter", "description": "Condensed matter physics"},
            {"code": "quant-ph", "name": "Quantum Physics", "description": "Quantum physics and quantum information"},
            {"code": "stat", "name": "Statistics", "description": "Statistics and statistical methods"},
            {"code": "q-bio", "name": "Quantitative Biology", "description": "Quantitative biology and bioinformatics"},
            {"code": "q-fin", "name": "Quantitative Finance", "description": "Quantitative finance and economics"},
            {"code": "econ", "name": "Economics", "description": "Economics and econometrics"}
        ],
        "search_tips": [
            "Use specific category codes like 'cs.LG' for machine learning",
            "Combine categories to find interdisciplinary research",
            "Add year ranges to focus on recent or historical work",
            "Include author names to find specific researchers",
            "Use methodology keywords like 'experimental' or 'theoretical'"
        ]
    }

# Add endpoint for search analytics
@app.get("/search-analytics")
async def search_analytics():
    """
    Get analytics about search patterns and database content
    """
    try:
        with get_db() as db:
            # Get basic statistics
            total_papers = db.execute(text("SELECT COUNT(*) FROM arxiv")).scalar()
            papers_with_embeddings = db.execute(text("SELECT COUNT(*) FROM arxiv WHERE embedding IS NOT NULL")).scalar()
            
            # Get category distribution
            category_query = text("""
                SELECT category, COUNT(*) as count
                FROM (
                    SELECT unnest(categories) as category
                    FROM arxiv
                ) cat_expanded
                GROUP BY category
                ORDER BY count DESC
                LIMIT 20
            """)
            
            category_results = db.execute(category_query).fetchall()
            category_distribution = {row.category: row.count for row in category_results}
            
            # Get temporal distribution
            temporal_query = text("""
                SELECT EXTRACT(YEAR FROM published_date) as year, COUNT(*) as count
                FROM arxiv
                WHERE published_date IS NOT NULL
                GROUP BY year
                ORDER BY year DESC
                LIMIT 20
            """)
            
            temporal_results = db.execute(temporal_query).fetchall()
            temporal_distribution = {int(row.year): row.count for row in temporal_results}
            
            return {
                "total_papers": total_papers,
                "papers_with_embeddings": papers_with_embeddings,
                "embedding_coverage": papers_with_embeddings / total_papers if total_papers > 0 else 0,
                "category_distribution": category_distribution,
                "temporal_distribution": temporal_distribution,
                "search_capabilities": {
                    "semantic_search": papers_with_embeddings > 0,
                    "category_hierarchy": True,
                    "temporal_analysis": True,
                    "author_analysis": True,
                    "collaboration_analysis": True
                }
            }
            
    except Exception as e:
        logger.error(f"Search analytics error: {str(e)}", exc_info=True)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True) 