# Research Assistant Chatbot

A powerful AI-powered research assistant that helps researchers find, analyze, and understand academic papers from arXiv.

## Features

### Chat Interface
- Interactive chat-based interface for research queries
- Context-aware conversations that maintain session history
- Support for HTML-formatted responses
- Session management with persistence

### Search Capabilities
- Advanced paper search with multiple filters:
  - Year-based filtering
  - Keyword search in titles and abstracts
  - Category-based filtering
- Real-time search results with detailed paper information
- Support for up to 50 results per search
- Comprehensive paper metadata including:
  - Title and authors
  - Abstract and categories
  - Publication and update dates
  - DOI and journal references

### Session Management
- Create and manage multiple chat sessions
- View session history with timestamps
- Update and delete sessions as needed
- Persistent storage of conversation context

## Tech Stack

### Backend
- FastAPI for high-performance API endpoints
- SQLAlchemy for database operations
- PostgreSQL for data storage
- Uvicorn as ASGI server
- Python 3.8+

### Frontend
- React for the user interface
- Tailwind CSS for styling
- Chart.js for data visualization
- Heroicons for UI icons

## API Endpoints

### Chat
- `POST /api/chat` - Process chat messages
- `GET /api/sessions` - List all chat sessions
- `GET /api/sessions/{session_id}` - Get specific session
- `DELETE /api/sessions/{session_id}` - Delete session
- `PUT /api/sessions/{session_id}` - Update session

### Search
- `POST /api/search` - Search for papers with filters
  - Supports year and keyword filters
  - Returns papers with full metadata
  - Includes summary of search results

### Health Check
- `GET /api/health` - Check API health status

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Node.js 16+
- Python 3.8+

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

3. Start the services:
```bash
docker-compose up --build
```

4. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

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
- FastAPI for the excellent web framework
- React and Tailwind CSS for the frontend tools 