# Math Papers Crawler

A Python application that crawls math-related papers from Semantic Scholar and stores them in a PostgreSQL database. The crawler saves the crawled data to JSON files for each run.

## Features

- Crawls math papers from Semantic Scholar API
- Stores papers in PostgreSQL database
- Saves crawled data to JSON files
- Configurable search parameters
- Rate limiting to respect API limits
- Docker-based setup for easy deployment

## Prerequisites

- Docker and Docker Compose
- Git (optional, for cloning the repository)

## Project Structure

```
math_crawler/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── session.py
│   ├── crawler/
│   │   ├── __init__.py
│   │   ├── scholar_crawler.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── logger.py
│   └── main.py
├── logs/
└── src/crawler/data/
```

## Setup Instructions

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd math_crawler
   ```

2. **Create environment file**:
   Copy the example environment file and modify it if needed:
   ```bash
   cp .env.example .env
   ```
   The default values in `.env` are:
   ```
   # Database settings
   DB_HOST=postgres
   DB_PORT=5432
   DB_NAME=scholar_db
   DB_USER=postgres
   DB_PASSWORD=postgres

   # Crawler settings
   MAX_RESULTS=5
   SEARCH_QUERY=mathematics
   DELAY_BETWEEN_REQUESTS=5.0
   ```

3. **Create required directories**:
   ```bash
   mkdir -p logs src/crawler/data
   ```

## Running the Application

1. **Start the services**:
   ```bash
   docker compose up --build -d
   ```
   This will:
   - Build the crawler service
   - Start PostgreSQL database
   - Start the crawler service

2. **Check the services status**:
   ```bash
   docker compose ps
   ```
   Both services should show as "running".

3. **View logs**:
   ```bash
   docker compose logs -f crawler
   ```
   This will show the crawler's progress in real-time.

4. **Check crawled data**:
   The crawled data will be saved in `src/crawler/data/` directory as JSON files with timestamps.

5. **Check database**:
   To view the papers stored in the database:
   ```bash
   docker compose exec postgres psql -U postgres -d scholar_db -c "SELECT title, year, citations FROM papers;"
   ```

## Stopping the Application

1. **Stop the services**:
   ```bash
   docker compose down
   ```

2. **Remove all data** (including database):
   ```bash
   docker compose down -v
   ```

## Configuration Options

You can modify the following settings in the `.env` file:

- `MAX_RESULTS`: Number of papers to crawl (default: 5)
- `SEARCH_QUERY`: Search term for papers (default: "mathematics")
- `DELAY_BETWEEN_REQUESTS`: Delay between API requests in seconds (default: 5.0)

## Data Storage

- **Database**: Papers are stored in PostgreSQL with the following schema:
  - id (SERIAL PRIMARY KEY)
  - title (TEXT, UNIQUE)
  - authors (TEXT)
  - abstract (TEXT)
  - year (INTEGER)
  - citations (INTEGER)
  - url (TEXT)
  - created_at (TIMESTAMP)

- **JSON Files**: Each crawl's data is saved in `src/crawler/data/` with format:
  ```json
  {
    "timestamp": "YYYYMMDD_HHMMSS",
    "total_papers_found": 5,
    "total_papers_stored": 5,
    "papers": [
      {
        "title": "...",
        "authors": "...",
        "abstract": "...",
        "year": 2023,
        "citations": 100,
        "url": "..."
      }
    ]
  }
  ```

## Troubleshooting

1. **Database connection issues**:
   - Check if PostgreSQL container is running: `docker compose ps`
   - Check PostgreSQL logs: `docker compose logs postgres`

2. **Crawler issues**:
   - Check crawler logs: `docker compose logs crawler`
   - Verify API rate limits and adjust `DELAY_BETWEEN_REQUESTS` if needed

3. **Data storage issues**:
   - Check if data directory exists: `ls src/crawler/data`
   - Verify database tables: `docker compose exec postgres psql -U postgres -d scholar_db -c "\dt"`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 