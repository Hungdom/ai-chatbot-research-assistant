import os
import logging
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from dotenv import load_dotenv
import requests
from tqdm import tqdm
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': '13.215.202.159',
    'port': os.getenv('DB_PORT', '5432'),
    'database': 'postgres',
    'user': 'postgres_kltn',
    'password': 'postgres_kltn2025_bngok'
}

# API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
API_MODEL = "text-embedding-3-small"

# Validate OpenAI API key
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

def get_text_for_embedding(row):
    """Combine relevant fields to create text for embedding"""
    text_parts = []
    
    # Add title
    if row['title']:
        text_parts.append(f"Title: {row['title']}")
    
    # Add authors (handle both list and JSON string formats)
    if row['authors']:
        try:
            if isinstance(row['authors'], str):
                authors = json.loads(row['authors'])
            else:
                authors = row['authors']
            
            if isinstance(authors, list):
                authors_text = ", ".join(authors)
                text_parts.append(f"Authors: {authors_text}")
            else:
                text_parts.append(f"Authors: {authors}")
        except (json.JSONDecodeError, TypeError):
            text_parts.append(f"Authors: {row['authors']}")
    
    # Add abstract
    if row['abstract']:
        text_parts.append(f"Abstract: {row['abstract']}")
    
    # Add categories (handle both list and JSON string formats)
    if row['categories']:
        try:
            if isinstance(row['categories'], str):
                categories = json.loads(row['categories'])
            else:
                categories = row['categories']
            
            if isinstance(categories, list):
                categories_text = ", ".join(categories)
                text_parts.append(f"Categories: {categories_text}")
            else:
                text_parts.append(f"Categories: {categories}")
        except (json.JSONDecodeError, TypeError):
            text_parts.append(f"Categories: {row['categories']}")
    
    # Add dates
    if row['published_date']:
        text_parts.append(f"Published: {row['published_date']}")
    if row['updated_date']:
        text_parts.append(f"Updated: {row['updated_date']}")
    
    # Add journal reference
    if row['journal_ref']:
        text_parts.append(f"Journal Reference: {row['journal_ref']}")
    
    # Add primary category
    if row['primary_category']:
        text_parts.append(f"Primary Category: {row['primary_category']}")
    
    # Add comment
    if row['comment']:
        text_parts.append(f"Comment: {row['comment']}")
    
    return " ".join(text_parts)

def get_embedding(text):
    """Get embedding from OpenAI API"""
    try:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": API_MODEL,
                "input": text
            }
        )
        response.raise_for_status()
        return response.json()['data'][0]['embedding']
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return None

def get_embeddings_batch(texts, api_batch_size=100):
    """Get embeddings for multiple texts in one API call"""
    try:
        # OpenAI supports up to 2048 inputs per request for embeddings
        all_embeddings = []
        
        for i in range(0, len(texts), api_batch_size):
            batch_texts = texts[i:i + api_batch_size]
            
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": API_MODEL,
                    "input": batch_texts
                }
            )
            response.raise_for_status()
            
            batch_embeddings = [item['embedding'] for item in response.json()['data']]
            all_embeddings.extend(batch_embeddings)
            
            # Rate limiting - be respectful to OpenAI
            time.sleep(0.1)  # Small delay between batches
            
        return all_embeddings
    except Exception as e:
        logger.error(f"Error getting batch embeddings: {str(e)}")
        return None

def update_embeddings(batch_size=1000):
    """Update embeddings for all rows in the arxiv table"""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                # Get total count
                # cur.execute("SELECT COUNT(*) FROM arxiv WHERE embedding IS NULL")
                # total_rows = cur.fetchone()[0]
                total_rows = 500000
                logger.info(f"Found {total_rows} rows to update")

                # Process in batches
                processed_records = 0
                batch_count = 0
                
                while processed_records < total_rows:
                    # Fetch batch of rows with all needed fields
                    cur.execute("""
                        SELECT id, title, abstract, authors, categories, 
                               published_date, updated_date, journal_ref, 
                               primary_category, comment
                        FROM arxiv 
                        WHERE embedding IS NULL and id > 500000
                        ORDER BY id 
                        LIMIT %s OFFSET %s
                    """, (batch_size, processed_records))
                    
                    rows = cur.fetchall()
                    if not rows:
                        break

                    # Prepare data for batch processing
                    batch_data = []
                    batch_texts = []
                    
                    for row in rows:
                        row_dict = {
                            'id': row[0],
                            'title': row[1],
                            'abstract': row[2],
                            'authors': row[3],
                            'categories': row[4],
                            'published_date': row[5],
                            'updated_date': row[6],
                            'journal_ref': row[7],
                            'primary_category': row[8],
                            'comment': row[9]
                        }
                        
                        # Get text for embedding
                        text = get_text_for_embedding(row_dict)
                        if text:
                            batch_data.append(row_dict)
                            batch_texts.append(text)
                    
                    # Process batch with OpenAI API
                    if batch_texts:
                        batch_count += 1
                        logger.info(f"Batch {batch_count}: Getting embeddings for {len(batch_texts)} texts...")
                        embeddings = get_embeddings_batch(batch_texts, api_batch_size=100)
                        
                        if embeddings and len(embeddings) == len(batch_data):
                            # Update database with batch results
                            updates = []
                            for i, (data, embedding) in enumerate(zip(batch_data, embeddings)):
                                updates.append((embedding, data['id']))
                            
                            # Batch update the database
                            cur.executemany("""
                                UPDATE arxiv 
                                SET embedding = %s 
                                WHERE id = %s
                            """, updates)
                            
                            logger.info(f"Updated {len(updates)} records with embeddings")
                        else:
                            logger.error(f"Mismatch in embeddings count: got {len(embeddings) if embeddings else 0}, expected {len(batch_data)}")
                    else:
                        logger.warning("No valid texts found in this batch")
                    
                    conn.commit()
                    processed_records += len(rows)
                    logger.info(f"Completed batch {batch_count}: Processed {min(processed_records, total_rows)}/{total_rows} rows")
                    

    except Exception as e:
        logger.error(f"Error updating embeddings: {str(e)}")
        raise

def main():
    logger.info("Starting embedding update process...")
    update_embeddings()
    logger.info("Embedding update process completed!")

if __name__ == "__main__":
    main() 