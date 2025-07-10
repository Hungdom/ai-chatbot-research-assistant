#!/usr/bin/env python3
"""
Script to generate embeddings for existing papers in the database.
This script should be run to populate embeddings for papers that don't have them.
"""

import asyncio
import os
import sys
import openai
from typing import List, Optional
from dotenv import load_dotenv
from database import get_db, Arxiv, init_db
from sqlalchemy import text
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

async def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generate embedding for the given text using OpenAI API
    """
    try:
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        # Clean and truncate text if needed
        cleaned_text = text.strip()
        if len(cleaned_text) > 8000:
            cleaned_text = cleaned_text[:8000]
        
        response = await openai.Embedding.acreate(
            input=cleaned_text,
            model="text-embedding-ada-002"
        )
        
        if response and response.data and len(response.data) > 0:
            return response.data[0].embedding
        else:
            logger.error("Invalid response from OpenAI embedding API")
            return None
            
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return None

async def generate_embeddings_for_papers(batch_size: int = 10, delay: float = 1.0):
    """
    Generate embeddings for papers that don't have them
    """
    try:
        logger.info("Starting embedding generation process...")
        
        # Check if OpenAI API key is configured
        if not openai.api_key:
            logger.error("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
            return False
        
        # Initialize database
        init_db()
        
        with get_db() as db:
            # Get count of papers without embeddings
            count_query = text("SELECT COUNT(*) FROM arxiv WHERE embedding IS NULL")
            total_papers = db.execute(count_query).scalar()
            
            if total_papers == 0:
                logger.info("All papers already have embeddings!")
                return True
            
            logger.info(f"Found {total_papers} papers without embeddings")
            
            # Process papers in batches
            processed = 0
            batch_num = 0
            
            while processed < total_papers:
                batch_num += 1
                logger.info(f"Processing batch {batch_num} (papers {processed+1}-{min(processed+batch_size, total_papers)})")
                
                # Get a batch of papers without embeddings
                batch_query = text("""
                    SELECT id, title, abstract 
                    FROM arxiv 
                    WHERE embedding IS NULL 
                    ORDER BY published_date DESC 
                    LIMIT :limit
                """)
                
                papers = db.execute(batch_query, {"limit": batch_size}).fetchall()
                
                if not papers:
                    logger.info("No more papers to process")
                    break
                
                # Generate embeddings for each paper in the batch
                for paper in papers:
                    paper_id, title, abstract = paper
                    
                    # Combine title and abstract for embedding
                    text_for_embedding = f"{title}\n\n{abstract}" if abstract else title
                    
                    logger.info(f"Generating embedding for paper {paper_id}: {title[:50]}...")
                    
                    # Generate embedding
                    embedding = await generate_embedding(text_for_embedding)
                    
                    if embedding:
                        # Convert embedding to proper format and update database
                        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                        
                        update_query = text("""
                            UPDATE arxiv 
                            SET embedding = :embedding::vector 
                            WHERE id = :paper_id
                        """)
                        
                        db.execute(update_query, {
                            "embedding": embedding_str,
                            "paper_id": paper_id
                        })
                        
                        logger.info(f"✓ Updated embedding for paper {paper_id}")
                    else:
                        logger.error(f"✗ Failed to generate embedding for paper {paper_id}")
                    
                    # Add delay to avoid rate limiting
                    await asyncio.sleep(delay)
                
                # Commit the batch
                db.commit()
                processed += len(papers)
                
                logger.info(f"Batch {batch_num} complete. Progress: {processed}/{total_papers} ({processed/total_papers*100:.1f}%)")
                
                # Longer delay between batches
                if processed < total_papers:
                    logger.info(f"Waiting {delay*2} seconds before next batch...")
                    await asyncio.sleep(delay * 2)
            
            logger.info("✓ Embedding generation complete!")
            
            # Final verification
            final_count = db.execute(count_query).scalar()
            logger.info(f"Papers without embeddings after processing: {final_count}")
            
            return True
            
    except Exception as e:
        logger.error(f"Error in embedding generation process: {str(e)}", exc_info=True)
        return False

async def main():
    """
    Main function to run the embedding generation
    """
    logger.info("=== ArXiv Paper Embedding Generator ===")
    
    # Check arguments
    batch_size = 10
    delay = 1.0
    
    if len(sys.argv) > 1:
        try:
            batch_size = int(sys.argv[1])
        except ValueError:
            logger.error("Invalid batch size argument")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            delay = float(sys.argv[2])
        except ValueError:
            logger.error("Invalid delay argument")
            sys.exit(1)
    
    logger.info(f"Using batch size: {batch_size}, delay: {delay}s")
    
    # Run the embedding generation
    success = await generate_embeddings_for_papers(batch_size, delay)
    
    if success:
        logger.info("✓ Embedding generation completed successfully!")
        sys.exit(0)
    else:
        logger.error("✗ Embedding generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 