# Research Assistant Chatbot

A powerful AI-powered research assistant that helps researchers find, analyze, and understand academic papers from arXiv with advanced semantic search, intelligent agents, and comprehensive data analysis capabilities.

## 🏗️ Architecture Overview

This system consists of multiple interconnected components:
- **Backend**: FastAPI server with intelligent agents
- **Frontend**: React-based user interface
- **Search Engine**: Elasticsearch-powered paper search
- **Database**: PostgreSQL with vector extensions
- **AI Agents**: Specialized agents for different tasks
- **Embedding System**: Local and cloud-based embeddings

## 🔧 Backend

### Core Technologies
- **FastAPI**: High-performance async web framework
- **SQLAlchemy**: Database ORM with PostgreSQL support
- **pgvector**: Vector similarity search for embeddings
- **OpenAI API**: GPT-4 for intelligent responses
- **Alembic**: Database migrations

### Key Components

#### Database Models (`database.py`)
```python
# ArXiv Papers Model
class Arxiv(Base):
    - arxiv_id: Unique paper identifier
    - title: Paper title with full-text search
    - abstract: Paper abstract
    - authors: Array of author names
    - categories: ArXiv categories
    - published_date: Publication date
    - embedding: 1536-dimensional vector for similarity search
    - doi: Digital Object Identifier
    - primary_category: Main research category
```

#### API Endpoints (`main.py`)
- `POST /api/chat`: Process chat messages with intelligent agents
- `GET /api/sessions`: List and manage chat sessions
- `POST /api/search`: Advanced paper search with filters
- `GET /api/health`: System health monitoring

#### Configuration
- Environment variables for database and API connections
- Connection pooling for optimal performance
- Logging configuration for debugging and monitoring

### Performance Features
- Connection pooling (5 connections, 10 max overflow)
- Query optimization with composite indexes
- Batch processing for embedding generation
- Async request handling

## 🤖 Agents

The system uses specialized AI agents for different tasks, each inheriting from `BaseAgent`:

### 1. Chat Agent (`chat_agent.py`)
**Purpose**: Main conversational interface for user interactions

**Key Features**:
- Intent analysis to understand user queries
- Context-aware conversations with session management
- Semantic paper search with embedding similarity
- Research insights generation
- Multi-modal response generation (text + HTML)

**Workflow**:
1. Analyze user intent using GPT-3.5-turbo
2. Search relevant papers using semantic similarity
3. Generate research insights and trends
4. Create comprehensive responses with citations
5. Update session history

### 2. Enhanced Chat Agent (`enhanced_chat_agent.py`)
**Purpose**: Advanced chat processing with multi-strategy intelligence

**Key Features**:
- Multi-strategy search approach (semantic, categorical, temporal)
- Enhanced intent analysis using GPT-4
- Research trend analysis and gap identification
- Author collaboration network analysis
- Comprehensive research landscape insights

**Advanced Capabilities**:
- Query complexity assessment
- Research area identification
- Temporal scope analysis
- Smart parameter extraction

### 3. Smart Search Agent (`smart_search_agent.py`)
**Purpose**: Intelligent paper discovery with advanced filtering

**Key Features**:
- Category hierarchy understanding
- Temporal pattern analysis
- Author-based search strategies
- Multi-dimensional relevance scoring
- Research trend identification

**Search Strategies**:
- Semantic similarity search
- Categorical expansion
- Temporal filtering
- Author collaboration analysis
- Hybrid approach combining multiple methods

### 4. Response Agent (`response_agent.py`)
**Purpose**: Specialized response generation and analysis

**Key Features**:
- Context-aware response generation
- Research trend analysis
- Yearly summary generation
- Category-specific insights
- No-results handling with suggestions

### 5. Base Agent (`base_agent.py`)
**Purpose**: Common functionality for all agents

**Shared Features**:
- OpenAI API integration
- Embedding generation
- Conversation history management
- Intent analysis utilities
- Error handling and logging

## 📊 Chatbot Dataflow

### 1. User Input Processing
```
User Query → Intent Analysis → Search Strategy Selection
```

### 2. Search Strategy Execution
```
Strategy Selection → {
    Semantic Search: Query Embedding → Similarity Search
    Categorical Search: Category Expansion → Filtered Search
    Temporal Search: Date Range → Time-based Filtering
    Author Search: Author Analysis → Collaboration Networks
    Hybrid Search: Combined Multi-strategy Approach
}
```

### 3. Result Processing
```
Raw Results → Relevance Scoring → Insight Generation → Response Formatting
```

### 4. Response Generation
```
Processed Results → {
    Text Response: GPT-4 Generated Analysis
    HTML Response: Structured Presentation
    Paper Citations: Formatted References
    Follow-up Questions: Suggested Queries
}
```

### 5. Session Management
```
Response → Session Storage → History Update → Context Preservation
```

### Data Flow Diagram
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   User      │───▶│   Intent    │───▶│   Search    │
│   Query     │    │   Analysis  │    │   Strategy  │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Response  │◀───│   Result    │◀───│   Paper     │
│   Generation│    │   Processing│    │   Retrieval │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 🎨 Frontend

### Technology Stack
- **React 18**: Modern functional components with hooks
- **Tailwind CSS**: Utility-first styling framework
- **Heroicons**: Beautiful SVG icons
- **Axios**: HTTP client for API communication
- **React Router**: Client-side routing
- **React Markdown**: Markdown rendering support

### Component Architecture

#### Pages (`src/pages/`)
1. **Home.js**: Landing page with features overview
2. **Chat.js**: Main chat interface with real-time messaging
3. **Search.js**: Advanced search interface with filters
4. **DatasetInsights.js**: Research analytics and visualizations

#### Components (`src/components/`)
1. **Layout.js**: Application shell with navigation
2. **ChatMessage.js**: Individual message rendering
3. **ChatSessions.js**: Session management interface
4. **ArxivCard.js**: Paper display component

### Key Features
- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **Real-time Chat**: WebSocket-like experience with polling
- **Session Management**: Persistent conversations
- **Search Filters**: Advanced filtering options
- **Paper Visualization**: Rich paper display with metadata
- **Markdown Support**: Formatted text rendering

### State Management
- React Context for global state
- Local state management with useState
- Session persistence with localStorage
- API state synchronization

## 🔍 Search Engine

### Architecture
- **Elasticsearch**: Full-text search and indexing
- **Docker Integration**: Containerized deployment
- **FastAPI Integration**: RESTful API endpoints
- **Cross-Origin Support**: CORS enabled for frontend

### Core Components (`search_engine/app/`)

#### Main Application (`main.py`)
```python
# Key Features:
- Elasticsearch connection with retry logic
- Date parsing with multiple formats
- Advanced search with filters
- Health monitoring
- CORS middleware
```

#### Search Capabilities
1. **Full-text Search**: Title and abstract searching
2. **Date Range Filtering**: Flexible date format support
3. **Category Filtering**: ArXiv category-based filtering
4. **Pagination**: Efficient result pagination
5. **Relevance Scoring**: Elasticsearch scoring algorithms

#### Supported Date Formats
- ISO formats: `YYYY-MM-DD`, `YYYY/MM/DD`
- US formats: `MM/DD/YYYY`, `MM-DD-YYYY`
- European formats: `DD/MM/YYYY`, `DD-MM-YYYY`
- Partial dates: `MM/YYYY`, `YYYY`

### Index Structure
```json
{
  "arxiv_index": {
    "mappings": {
      "properties": {
        "title": {"type": "text", "analyzer": "standard"},
        "abstract": {"type": "text", "analyzer": "standard"},
        "authors": {"type": "keyword"},
        "categories": {"type": "keyword"},
        "published_date": {"type": "date"},
        "doi": {"type": "keyword"}
      }
    }
  }
}
```

### Performance Optimization
- Connection pooling with retry logic
- Request timeout configuration
- Efficient query construction
- Result caching strategies

## 🗄️ Postgres Data

### Database Architecture
- **PostgreSQL 13+**: Main database system
- **pgvector Extension**: Vector similarity search
- **Connection Pooling**: Optimized connection management
- **Indexing Strategy**: Composite indexes for query optimization

### Schema Design

#### ArXiv Papers Table
```sql
CREATE TABLE arxiv (
    id SERIAL PRIMARY KEY,
    arxiv_id VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    abstract TEXT,
    authors TEXT[],
    categories TEXT[],
    published_date TIMESTAMP,
    updated_date TIMESTAMP,
    doi VARCHAR(100),
    primary_category VARCHAR(50),
    comment TEXT,
    embedding VECTOR(1536)  -- OpenAI embeddings
);
```

#### Chat Sessions Table
```sql
CREATE TABLE chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) UNIQUE NOT NULL,
    messages JSONB DEFAULT '[]',
    context JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Indexing Strategy
```sql
-- Performance indexes
CREATE INDEX idx_arxiv_published_date ON arxiv(published_date);
CREATE INDEX idx_arxiv_primary_category ON arxiv(primary_category);
CREATE INDEX idx_arxiv_title_gin ON arxiv USING gin(to_tsvector('english', title));
CREATE INDEX idx_arxiv_abstract_gin ON arxiv USING gin(to_tsvector('english', abstract));
CREATE INDEX idx_arxiv_categories_gin ON arxiv USING gin(categories);

-- Vector similarity index
CREATE INDEX idx_arxiv_embedding ON arxiv USING hnsw (embedding vector_cosine_ops);
```

### Vector Operations
```sql
-- Similarity search example
SELECT arxiv_id, title, 
       cosine_similarity(embedding, $1::vector) as similarity
FROM arxiv 
WHERE embedding IS NOT NULL 
ORDER BY similarity DESC 
LIMIT 10;
```

### Data Migration
- **Alembic**: Database schema versioning
- **Migration Scripts**: Automated schema updates
- **Data Validation**: Integrity checks and constraints
- **Backup Strategy**: Point-in-time recovery

### Performance Optimization
- **Connection Pooling**: 5 connections, 10 max overflow
- **Query Optimization**: Composite indexes
- **Vacuum Strategy**: Automated maintenance
- **Partitioning**: For large datasets (future enhancement)

## 📈 Embedding System

### Local Embeddings (`embedding/`)
- **CPU Optimized**: Efficient local processing
- **GPU Support**: CUDA acceleration when available
- **Batch Processing**: Optimized throughput
- **Memory Management**: Efficient resource usage

### Cloud Embeddings
- **OpenAI API**: text-embedding-ada-002 model
- **Rate Limiting**: API usage optimization
- **Fallback Strategy**: Local processing when API unavailable
- **Cost Optimization**: Batch processing and caching

## Features

### Chat Interface
- Interactive chat-based interface for research queries
- Context-aware conversations that maintain session history
- Support for HTML-formatted responses
- Session management with persistence
- Multi-agent processing for intelligent responses

### Search Capabilities
- Advanced paper search with multiple filters:
  - Year-based filtering
  - Keyword search in titles and abstracts
  - Category-based filtering
  - Semantic similarity search
- Real-time search results with detailed paper information
- Support for up to 50 results per search
- Comprehensive paper metadata including:
  - Title and authors
  - Abstract and categories
  - Publication and update dates
  - DOI and journal references
  - Similarity scores for semantic search

### Session Management
- Create and manage multiple chat sessions
- View session history with timestamps
- Update and delete sessions as needed
- Persistent storage of conversation context

### AI-Powered Features
- Intent analysis and query understanding
- Research trend identification
- Author collaboration analysis
- Research gap identification
- Follow-up question generation

## Tech Stack

### Backend
- FastAPI for high-performance API endpoints
- SQLAlchemy for database operations
- PostgreSQL with pgvector for vector similarity
- OpenAI API for intelligent responses
- Elasticsearch for full-text search
- Uvicorn as ASGI server
- Python 3.8+

### Frontend
- React 18 for the user interface
- Tailwind CSS for styling
- Heroicons for UI icons
- Axios for API communication
- React Router for navigation

### Infrastructure
- Docker and Docker Compose for containerization
- Elasticsearch for search indexing
- PostgreSQL for data storage
- Redis for caching (optional)

## API Endpoints

### Chat
- `POST /api/chat` - Process chat messages with AI agents
- `GET /api/sessions` - List all chat sessions
- `GET /api/sessions/{session_id}` - Get specific session
- `DELETE /api/sessions/{session_id}` - Delete session
- `PUT /api/sessions/{session_id}` - Update session

### Search
- `POST /api/search` - Search for papers with filters
  - Supports year and keyword filters
  - Returns papers with full metadata
  - Includes similarity scores
  - Provides search result summaries

### Health Check
- `GET /api/health` - Check API health status

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Node.js 16+
- Python 3.8+
- PostgreSQL 13+
- Elasticsearch 8+

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd research-assistant-chatbot
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Initialize the database:
```bash
cd backend
python init_db.py
```

4. Generate embeddings (optional but recommended):
```bash
python generate_embeddings.py
```

5. Start the services:
```bash
docker-compose up --build
```

6. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Search Engine: http://localhost:9200

## Development

### Backend Development
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Development
```bash
cd frontend
npm install
npm start
```

### Search Engine Development
```bash
cd search_engine
docker-compose up elasticsearch
python app/main.py
```

## Advanced Features

### Embedding Generation
For optimal search results, generate embeddings for your papers:
```bash
cd backend
python generate_embeddings.py [batch_size] [delay]
```

### Enhanced Search
The system includes advanced search capabilities:
- Category hierarchy understanding
- Temporal pattern analysis
- Author collaboration networks
- Multi-strategy search approaches

See `backend/ENHANCED_SEARCH_GUIDE.md` for detailed information.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- arXiv for providing the research paper dataset
- OpenAI for GPT-4 and embedding services
- FastAPI for the excellent web framework
- React and Tailwind CSS for the frontend tools
- PostgreSQL and pgvector for vector similarity search
- Elasticsearch for full-text search capabilities 