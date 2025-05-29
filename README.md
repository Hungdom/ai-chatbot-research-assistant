# Research Assistant Chat Bot

A sophisticated research assistant application that helps users find and analyze academic papers using natural language queries. The system combines the power of GPT-4 with a PostgreSQL database to provide intelligent responses about research papers.

## Architecture

The project follows a modern microservices architecture with the following components:

### Backend (FastAPI)
- **API Layer**: Built with FastAPI, providing RESTful endpoints for chat interactions
- **Database Layer**: PostgreSQL database with SQLAlchemy ORM
- **AI Integration**: OpenAI GPT-4 for natural language processing and response generation
- **CORS Support**: Configured for frontend integration

### Frontend (React)
- Modern web interface for user interactions
- Real-time chat interface
- Filtering capabilities for papers

## Components

### Database Schema
The system uses a PostgreSQL database with the following main table:

#### Papers Table
- `id`: Primary key
- `title`: Paper title
- `abstract`: Paper abstract
- `authors`: Array of author names
- `keywords`: Array of keywords
- `year`: Publication year

### API Endpoints

1. **POST /chat**
   - Main endpoint for chat interactions
   - Accepts queries with optional filters
   - Returns AI-generated responses with relevant papers

2. **GET /**
   - Health check endpoint
   - Returns API status

## Setup Instructions

### Prerequisites
- Python 3.8+
- PostgreSQL
- Node.js and npm
- OpenAI API key

### Environment Variables
Create a `.env` file in the backend directory with:
```
DATABASE_URL=postgresql://postgres:postgres@postgres_db:5432/research_db
OPENAI_API_KEY=your_openai_api_key
```

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Features

- Natural language query processing
- Paper filtering by year and keywords
- AI-powered responses based on paper content
- Real-time chat interface
- Paper metadata display
- CORS support for local development

## API Usage

### Chat Endpoint
```python
POST /chat
{
    "query": "What are the latest developments in quantum computing?",
    "year_filter": 2023,  # Optional
    "keywords": ["quantum", "computing"]  # Optional
}
```

Response:
```python
{
    "response": "AI-generated response...",
    "papers": [
        {
            "title": "Paper title",
            "abstract": "Paper abstract",
            "authors": ["Author 1", "Author 2"],
            "year": 2023,
            "keywords": ["keyword1", "keyword2"]
        }
    ]
}
```

## Development

### Adding New Features
1. Backend changes should be made in the `backend` directory
2. Frontend changes should be made in the `frontend` directory
3. Update the database schema in `backend/main.py`
4. Add new API endpoints as needed

### Testing
- Backend tests can be added in the `backend/tests` directory
- Frontend tests can be added in the `frontend/src/tests` directory

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details. 