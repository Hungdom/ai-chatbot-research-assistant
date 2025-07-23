import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'scholar_db'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
}

# Crawler configuration
CRAWLER_CONFIG = {
    'year_range': {
        'start': 1900,
        'end': 2024
    },
    'delay_between_requests': 2.0,
    'max_retries': 3
}

# Math topics to crawl
MATH_TOPICS = [
    'Algebra',
    'Analysis',
    'Applied Mathematics',
    'Combinatorics',
    'Geometry',
    'Logic',
    'Number Theory',
    'Probability',
    'Statistics',
    'Topology'
] 