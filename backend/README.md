# Research Assistant Chatbot Backend

A sophisticated AI-powered research assistant backend that provides intelligent search, analysis, and chat capabilities for academic papers from ArXiv.

## 🚀 Overview

This backend provides a comprehensive research assistant platform with the following key features:

- **Intelligent Chat Interface**: Multi-agent conversational AI system
- **Advanced Search Capabilities**: Semantic search using vector embeddings
- **Research Analysis**: Automated paper analysis and trend identification
- **Session Management**: Persistent chat sessions with context awareness
- **Real-time Analytics**: Search analytics and category insights

## 🏗️ Architecture

### Core Components

```
backend/
├── agents/                    # AI Agent System
│   ├── base_agent.py         # Base agent class with common functionality
│   ├── chat_agent.py         # Main conversational agent
│   ├── enhanced_chat_agent.py # Enhanced chat with advanced features
│   ├── smart_search_agent.py # Intelligent search agent
│   └── response_agent.py     # Response generation agent
├── config/                   # Configuration Files
│   └── prompts.py           # AI prompts and system messages
├── migrations/              # Database Migration Scripts
│   ├── env.py              # Alembic environment configuration
│   └── script.py.mako      # Migration script template
├── routes/                  # API Routes (Future expansion)
│   └── search.js           # Search route implementation
├── database.py             # Database models and connection
├── main.py                # FastAPI application entry point
└── requirements.txt       # Python dependencies
```

### Database Schema

- **Arxiv Table**: Stores academic papers with vector embeddings
- **ChatSession Table**: Manages conversation sessions and context
- **Vector Search**: Utilizes PostgreSQL with pgvector extension

### AI Agent System

The backend uses a multi-agent architecture:

1. **BaseAgent**: Foundation class with OpenAI integration
2. **ChatAgent**: Handles conversational interactions
3. **EnhancedChatAgent**: Advanced chat with context awareness
4. **SmartSearchAgent**: Intelligent paper search and filtering
5. **ResponseAgent**: Generates structured responses

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL 12+ with pgvector extension
- OpenAI API key
- Node.js 14+ (for some route handlers)

## 🛠️ Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Install PostgreSQL and pgvector extension
# For Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib
sudo -u postgres psql -c "CREATE EXTENSION vector;"

# Create database
createdb research_assistant

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 3. Environment Variables

Create a `.env` file in the backend directory:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=research_assistant
DB_USER=postgres
DB_PASSWORD=your_password

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Application Configuration
DEBUG=True
LOG_LEVEL=INFO
```

### 4. Database Initialization

```bash
# Run database migrations
alembic upgrade head

# Initialize database tables
python init_db.py

# Generate embeddings for existing papers (if any)
python generate_embeddings.py
```

## 🚀 Running the Application

### Development Mode

```bash
# Start the FastAPI server
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
# Using Docker
docker build -t research-assistant-backend .
docker run -p 8000:8000 research-assistant-backend

# Or using uvicorn with production settings
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📡 API Endpoints

### Chat Endpoints

#### POST `/api/chat`
Main chat endpoint for conversational interactions.

**Request:**
```json
{
  "query": "What are the latest developments in machine learning?",
  "session_id": "optional-session-id",
  "context": {}
}
```

**Response:**
```json
{
  "response": "Based on recent research...",
  "html_response": "<p>Based on recent research...</p>",
  "arxiv": [...],
  "intent": {...},
  "session_id": "session-uuid",
  "metadata": {...}
}
```

#### POST `/enhanced-chat`
Enhanced chat with advanced AI capabilities.

#### POST `/chat`
Alternative chat endpoint with different processing.

### Search Endpoints

#### POST `/api/search`
Search for academic papers with filters.

**Request:**
```json
{
  "year": 2023,
  "keywords": ["machine learning", "neural networks"]
}
```

**Response:**
```json
{
  "papers": [...],
  "summary": "Found 42 papers related to..."
}
```

#### POST `/smart-search`
Intelligent search using AI agents.

#### POST `/research-landscape`
Comprehensive research landscape analysis.

### Session Management

#### GET `/api/sessions`
Retrieve all chat sessions.

#### GET `/api/sessions/{session_id}`
Get specific session details.

#### DELETE `/api/sessions/{session_id}`
Delete a chat session.

#### PUT `/api/sessions/{session_id}`
Update session data.

### Analytics Endpoints

#### GET `/category-analysis/{category}`
Analyze papers by category.

#### GET `/category-suggestions`
Get category suggestions.

#### GET `/search-analytics`
Retrieve search analytics data.

### Health Check

#### GET `/api/health`
Health check endpoint.

#### GET `/`
Root endpoint with API information.

## 🔧 Configuration

### Prompts Configuration

The `config/prompts.py` file contains all AI system prompts:

- **REQUEST_AGENT_PROMPTS**: Query analysis and follow-up generation
- **QUERY_AGENT_PROMPTS**: Search strategy determination
- **RESPONSE_AGENT_PROMPTS**: Response analysis and formatting

### Database Configuration

Database models are defined in `database.py`:

- **Arxiv Model**: Academic paper storage with vector embeddings
- **ChatSession Model**: Session management with JSONB storage

## 🔍 Search System

The backend implements a sophisticated search system:

### Vector Search
- Uses OpenAI embeddings (text-embedding-ada-002)
- PostgreSQL with pgvector extension
- Cosine similarity for relevance ranking

### Hybrid Search Strategy
- Semantic search using embeddings
- Category-based hierarchical search
- Keyword matching with TF-IDF
- Temporal and author-based filtering

### Search Features
- Smart fallback when embeddings unavailable
- Category hierarchy expansion
- Research landscape analysis
- Trend identification

## 🧪 Testing

### Unit Tests

```bash
# Run similarity tests
python test_similarity.py

# Run debug tests
python debug_test.py

# Run smart search examples
python smart_search_examples.py
```

### Test Files

- `test_similarity.py`: Vector similarity testing
- `debug_test.py`: Debug utilities and tests
- `smart_search_examples.py`: Search functionality examples

## 📊 Database Management

### Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback migrations
alembic downgrade -1
```

### Embedding Generation

```bash
# Generate embeddings for all papers
python generate_embeddings.py

# Generate embeddings with custom batch size
python generate_embeddings.py --batch_size 50 --delay 1.0
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build Docker image
docker build -t research-assistant-backend .

# Run container
docker run -p 8000:8000 \
  -e DB_HOST=host.docker.internal \
  -e OPENAI_API_KEY=your_key \
  research-assistant-backend
```

### Docker Compose

```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=postgres
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
  
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      - POSTGRES_DB=research_assistant
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## 📝 Logging

The application uses Python's logging module:

- **Level**: INFO by default
- **Format**: Timestamped with module names
- **Output**: Console and file (configurable)

## 🔒 Security

### API Security
- CORS configuration for cross-origin requests
- Input validation using Pydantic models
- SQL injection prevention with SQLAlchemy ORM

### Environment Security
- Sensitive data in environment variables
- API keys never hardcoded
- Database credentials secured

## 🚨 Troubleshooting

### Common Issues

1. **0% Similarity Results**
   - Run `python generate_embeddings.py` to generate embeddings
   - Check OpenAI API key configuration
   - Verify pgvector extension is installed

2. **Database Connection Issues**
   - Verify PostgreSQL is running
   - Check database credentials in `.env`
   - Ensure database exists and is accessible

3. **OpenAI API Errors**
   - Verify API key is valid and has credits
   - Check rate limits and usage
   - Ensure correct model names are used

### Debug Mode

Enable debug mode by setting `DEBUG=True` in `.env`:

```bash
# This enables SQL query logging and detailed error messages
DEBUG=True
LOG_LEVEL=DEBUG
```

## 📚 Additional Documentation

- [`ENHANCED_SEARCH_GUIDE.md`](./ENHANCED_SEARCH_GUIDE.md): Detailed search system documentation
- [`README_EMBEDDINGS.md`](./README_EMBEDDINGS.md): Embedding generation guide
- [`smart_search_examples.py`](./smart_search_examples.py): Search implementation examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- Check the troubleshooting section
- Review existing documentation
- Open an issue on GitHub
- Contact the development team 