import os
import logging
import psycopg2
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import time
import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import pickle

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

# Model configuration
TARGET_DIMENSIONS = 1536  # Match OpenAI text-embedding-3-small standard
MODEL_NAME = "BAAI/bge-large-en-v1.5"  # 1024 dimensions originally
# Alternative models:
# "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions
# "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
# "intfloat/e5-large-v2"  # 1024 dimensions

class LocalEmbedder:
    def __init__(self, model_name=MODEL_NAME, target_dim=TARGET_DIMENSIONS):
        self.model_name = model_name
        self.target_dim = target_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.original_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Original dimensions: {self.original_dim}, Target: {target_dim}")
        
        # Initialize dimensionality reducer
        self.reducer = None
        self.is_fitted = False
        
        # Load pre-fitted reducer if it exists
        self.load_reducer()
    
    def load_reducer(self):
        """Load pre-fitted PCA reducer if available"""
        reducer_path = f"pca_reducer_{self.model_name.replace('/', '_')}_{self.target_dim}.pkl"
        if os.path.exists(reducer_path):
            try:
                with open(reducer_path, 'rb') as f:
                    self.reducer = pickle.load(f)
                self.is_fitted = True
                logger.info(f"Loaded pre-fitted PCA reducer from {reducer_path}")
            except Exception as e:
                logger.warning(f"Failed to load reducer: {e}")
    
    def save_reducer(self):
        """Save fitted PCA reducer"""
        if self.reducer and self.is_fitted:
            reducer_path = f"pca_reducer_{self.model_name.replace('/', '_')}_{self.target_dim}.pkl"
            try:
                with open(reducer_path, 'wb') as f:
                    pickle.dump(self.reducer, f)
                logger.info(f"Saved PCA reducer to {reducer_path}")
            except Exception as e:
                logger.error(f"Failed to save reducer: {e}")
    
    def fit_reducer(self, sample_texts, sample_size=10000):
        """Fit PCA reducer on a sample of texts"""
        if self.is_fitted:
            logger.info("Reducer already fitted, skipping...")
            return
        
        logger.info(f"Fitting PCA reducer on {min(len(sample_texts), sample_size)} samples...")
        
        # Take a sample for fitting
        if len(sample_texts) > sample_size:
            import random
            sample_texts = random.sample(sample_texts, sample_size)
        
        # Get embeddings for sample
        embeddings = self.model.encode(sample_texts, show_progress_bar=True, batch_size=32)
        
        # Fit PCA
        if self.target_dim < self.original_dim:
            from sklearn.decomposition import PCA
            self.reducer = PCA(n_components=self.target_dim, random_state=42)
        else:
            # If target dim is larger, we'll use zero-padding
            self.reducer = None
        
        if self.reducer:
            logger.info(f"Fitting PCA to reduce from {self.original_dim} to {self.target_dim} dimensions...")
            self.reducer.fit(embeddings)
            explained_variance = np.sum(self.reducer.explained_variance_ratio_)
            logger.info(f"PCA fitted. Explained variance ratio: {explained_variance:.4f}")
        
        self.is_fitted = True
        self.save_reducer()
    
    def get_embeddings(self, texts, batch_size=32):
        """Get embeddings with target dimensions"""
        if not self.is_fitted and self.target_dim != self.original_dim:
            logger.warning("Reducer not fitted! Fitting on current batch...")
            self.fit_reducer(texts[:1000])  # Fit on first 1000 texts
        
        # Get original embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=batch_size)
        
        # Transform to target dimensions
        if self.target_dim == self.original_dim:
            return embeddings
        elif self.target_dim < self.original_dim and self.reducer:
            # Reduce dimensions with PCA
            return self.reducer.transform(embeddings)
        elif self.target_dim > self.original_dim:
            # Pad with zeros
            padding = np.zeros((embeddings.shape[0], self.target_dim - self.original_dim))
            return np.hstack([embeddings, padding])
        else:
            # Truncate if no reducer available
            return embeddings[:, :self.target_dim]

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

def update_embeddings(batch_size=100, embedding_batch_size=32):
    """Update embeddings for all rows in the arxiv table using local model"""
    
    # Initialize local embedder
    embedder = LocalEmbedder()
    
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                # Get total count
                total_rows = 500000
                logger.info(f"Found {total_rows} rows to update")

                # Skip PCA fitting since we're expanding dimensions (1024 -> 1536), just use zero-padding
                if TARGET_DIMENSIONS > embedder.original_dim:
                    logger.info("Target dimensions > original dimensions, using zero-padding (no PCA needed)")
                    embedder.is_fitted = True
                    embedder.reducer = None
                elif not embedder.is_fitted and TARGET_DIMENSIONS != embedder.original_dim:
                    logger.info("Collecting sample texts for PCA fitting...")
                    cur.execute("""
                        SELECT title, abstract, authors, categories, 
                               published_date, updated_date, journal_ref, 
                               primary_category, comment
                        FROM arxiv 
                        WHERE embedding IS NULL and id > 500000
                        ORDER BY id 
                        LIMIT 1000
                    """)
                    
                    sample_rows = cur.fetchall()
                    sample_texts = []
                    
                    for row in sample_rows:
                        row_dict = {
                            'title': row[0],
                            'abstract': row[1],
                            'authors': row[2],
                            'categories': row[3],
                            'published_date': row[4],
                            'updated_date': row[5],
                            'journal_ref': row[6],
                            'primary_category': row[7],
                            'comment': row[8]
                        }
                        text = get_text_for_embedding(row_dict)
                        if text:
                            sample_texts.append(text)
                    
                    embedder.fit_reducer(sample_texts)

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
                    
                    # Process batch with local model
                    if batch_texts:
                        batch_count += 1
                        logger.info(f"Batch {batch_count}: Getting embeddings for {len(batch_texts)} texts...")
                        
                        # Get embeddings using local model
                        embeddings = embedder.get_embeddings(batch_texts, batch_size=embedding_batch_size)
                        
                        if embeddings is not None and len(embeddings) == len(batch_data):
                            # Convert to list format for PostgreSQL
                            embeddings_list = embeddings.tolist()
                            
                            # Update database with batch results
                            updates = []
                            for i, (data, embedding) in enumerate(zip(batch_data, embeddings_list)):
                                updates.append((embedding, data['id']))
                            
                            # Batch update the database
                            cur.executemany("""
                                UPDATE arxiv 
                                SET embedding = %s 
                                WHERE id = %s
                            """, updates)
                            
                            # Commit immediately after each batch
                            conn.commit()
                            
                            logger.info(f"✅ BATCH {batch_count} COMPLETED: Updated {len(updates)} records with {TARGET_DIMENSIONS}D embeddings and COMMITTED to database")
                        else:
                            logger.error(f"Mismatch in embeddings count: got {len(embeddings) if embeddings is not None else 0}, expected {len(batch_data)}")
                    else:
                        logger.warning("No valid texts found in this batch")
                    
                    processed_records += len(rows)
                    logger.info(f"📊 PROGRESS: Processed {min(processed_records, total_rows)}/{total_rows} rows ({100*min(processed_records, total_rows)/total_rows:.1f}%)")
                    
                    # Memory cleanup
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        logger.error(f"Error updating embeddings: {str(e)}")
        raise

def test_embeddings():
    """Test the local embedding setup"""
    logger.info("Testing local embeddings...")
    
    embedder = LocalEmbedder()
    
    test_texts = [
        "This is a test document about machine learning.",
        "Another document discussing natural language processing.",
        "A paper on computer vision and deep learning."
    ]
    
    # Fit reducer if needed
    embedder.fit_reducer(test_texts)
    
    # Get embeddings
    embeddings = embedder.get_embeddings(test_texts)
    
    logger.info(f"Test successful!")
    logger.info(f"Input texts: {len(test_texts)}")
    logger.info(f"Output embeddings shape: {embeddings.shape}")
    logger.info(f"Target dimensions: {TARGET_DIMENSIONS}")
    logger.info(f"Actual dimensions: {embeddings.shape[1]}")
    
    return embeddings.shape[1] == TARGET_DIMENSIONS

def main():
    logger.info("Starting local embedding process...")
    
    # Test setup first
    if test_embeddings():
        logger.info("✅ Local embedding test passed!")
        logger.info("Starting batch processing...")
        update_embeddings()
        logger.info("✅ Local embedding process completed!")
    else:
        logger.error("❌ Local embedding test failed!")

if __name__ == "__main__":
    main() 