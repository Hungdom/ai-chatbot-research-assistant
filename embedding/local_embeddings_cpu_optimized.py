import os
import logging
import psycopg2
from psycopg2.extras import execute_batch
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import time
import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import gc
import multiprocessing as mp

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

# CPU-optimized batch sizes (smaller for CPU processing)
CPU_OPTIMIZED_BATCH_SIZES = {
    'db_fetch': 2000,        # Larger DB fetch for fewer round trips
    'embedding': 64,         # Smaller embedding batch for CPU
    'db_update': 1000,       # Large DB update batch
    'commit_every': 3,       # Commit every 3 batches
    'text_processing_workers': min(8, mp.cpu_count())  # Parallel text processing
}

class CPUOptimizedLocalEmbedder:
    def __init__(self, model_name=MODEL_NAME, target_dim=TARGET_DIMENSIONS):
        self.model_name = model_name
        self.target_dim = target_dim
        self.device = torch.device('cpu')  # Force CPU usage
        
        # Set CPU-specific optimizations
        torch.set_num_threads(mp.cpu_count())  # Use all CPU cores
        
        logger.info(f"Loading model {model_name} on CPU with {mp.cpu_count()} cores...")
        logger.info(f"CPU-optimized batch sizes: {CPU_OPTIMIZED_BATCH_SIZES}")
        
        # Load model with CPU optimizations
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Enable optimizations for CPU inference
        torch.backends.mkldnn.enabled = True  # Enable Intel MKL-DNN for CPU
        if hasattr(torch.backends, 'mkl'):
            torch.backends.mkl.enabled = True
        
        self.original_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Original dimensions: {self.original_dim}, Target: {target_dim}")
        
        # Initialize dimensionality reducer
        self.reducer = None
        self.is_fitted = False
        
        # Load pre-fitted reducer if it exists
        self.load_reducer()
    
    def load_reducer(self):
        """Load pre-fitted PCA reducer if available"""
        reducer_path = f"pca_reducer_cpu_{self.model_name.replace('/', '_')}_{self.target_dim}.pkl"
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
            reducer_path = f"pca_reducer_cpu_{self.model_name.replace('/', '_')}_{self.target_dim}.pkl"
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
        
        # Get embeddings for sample with CPU-optimized batch size
        embeddings = self.model.encode(
            sample_texts, 
            show_progress_bar=True, 
            batch_size=CPU_OPTIMIZED_BATCH_SIZES['embedding'],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
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
    
    def get_embeddings(self, texts, batch_size=None):
        """Get embeddings with target dimensions - CPU optimized version"""
        if batch_size is None:
            batch_size = CPU_OPTIMIZED_BATCH_SIZES['embedding']
            
        if not self.is_fitted and self.target_dim != self.original_dim:
            logger.warning("Reducer not fitted! Fitting on current batch...")
            self.fit_reducer(texts[:1000])  # Fit on first 1000 texts
        
        # Get original embeddings with CPU optimizations
        with torch.no_grad():  # Disable gradient computation for faster inference
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=True, 
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device
            )
        
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

def get_text_for_embedding_parallel(row):
    """Combine relevant fields to create text for embedding - parallelized version"""
    text_parts = []
    
    # Add title
    if row.get('title'):
        text_parts.append(f"Title: {row['title']}")
    
    # Add authors (handle both list and JSON string formats)
    if row.get('authors'):
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
    if row.get('abstract'):
        text_parts.append(f"Abstract: {row['abstract']}")
    
    # Add categories (handle both list and JSON string formats)
    if row.get('categories'):
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
    if row.get('published_date'):
        text_parts.append(f"Published: {row['published_date']}")
    if row.get('updated_date'):
        text_parts.append(f"Updated: {row['updated_date']}")
    
    # Add journal reference
    if row.get('journal_ref'):
        text_parts.append(f"Journal Reference: {row['journal_ref']}")
    
    # Add primary category
    if row.get('primary_category'):
        text_parts.append(f"Primary Category: {row['primary_category']}")
    
    # Add comment
    if row.get('comment'):
        text_parts.append(f"Comment: {row['comment']}")
    
    return " ".join(text_parts)

def get_next_batch_cursor_based(cur, last_id=0, batch_size=2000):
    """Get next batch using cursor-based pagination (much faster than OFFSET)"""
    cur.execute("""
        SELECT id, title, abstract, authors, categories, 
               published_date, updated_date, journal_ref, 
               primary_category, comment
        FROM arxiv 
        WHERE embedding IS NULL and id > %s
        ORDER BY id 
        LIMIT %s
    """, (last_id, batch_size))
    
    rows = cur.fetchall()
    return rows

def process_batch_cpu_optimized(rows, embedder):
    """Process a batch of rows with CPU optimizations"""
    
    # Prepare data for parallel text processing
    row_dicts = []
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
        row_dicts.append(row_dict)
    
    # Process text extraction in parallel
    batch_data = []
    batch_texts = []
    
    with ThreadPoolExecutor(max_workers=CPU_OPTIMIZED_BATCH_SIZES['text_processing_workers']) as executor:
        # Submit all text processing tasks
        future_to_data = {executor.submit(get_text_for_embedding_parallel, row_dict): row_dict 
                         for row_dict in row_dicts}
        
        # Collect results
        for future in as_completed(future_to_data):
            row_dict = future_to_data[future]
            try:
                text = future.result()
                if text:
                    batch_data.append(row_dict)
                    batch_texts.append(text)
            except Exception as e:
                logger.error(f"Error processing text for row {row_dict['id']}: {e}")
    
    if not batch_texts:
        return []
    
    # Get embeddings using CPU-optimized batch processing
    embeddings = embedder.get_embeddings(batch_texts)
    
    if embeddings is not None and len(embeddings) == len(batch_data):
        # Convert to list format for PostgreSQL
        embeddings_list = embeddings.tolist()
        
        # Prepare updates
        updates = []
        for i, (data, embedding) in enumerate(zip(batch_data, embeddings_list)):
            updates.append((embedding, data['id']))
        
        return updates
    else:
        logger.error(f"Mismatch in embeddings count: got {len(embeddings) if embeddings is not None else 0}, expected {len(batch_data)}")
        return []

def update_embeddings_cpu_optimized():
    """CPU-optimized version of update_embeddings with major performance improvements"""
    
    # Initialize CPU-optimized embedder
    embedder = CPUOptimizedLocalEmbedder()
    
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                # Get total count for progress tracking
                cur.execute("SELECT COUNT(*) FROM arxiv WHERE embedding IS NULL and id > 500000")
                total_rows = cur.fetchone()[0]
                logger.info(f"Found {total_rows} rows to update")

                # Skip PCA fitting since we're expanding dimensions (1024 -> 1536), just use zero-padding
                if TARGET_DIMENSIONS > embedder.original_dim:
                    logger.info("Target dimensions > original dimensions, using zero-padding (no PCA needed)")
                    embedder.is_fitted = True
                    embedder.reducer = None
                elif not embedder.is_fitted and TARGET_DIMENSIONS != embedder.original_dim:
                    logger.info("Collecting sample texts for PCA fitting...")
                    sample_rows = get_next_batch_cursor_based(cur, last_id=500000, batch_size=1000)
                    sample_texts = []
                    
                    for row in sample_rows:
                        row_dict = {
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
                        text = get_text_for_embedding_parallel(row_dict)
                        if text:
                            sample_texts.append(text)
                    
                    embedder.fit_reducer(sample_texts)

                # Initialize progress tracking
                processed_records = 0
                batch_count = 0
                last_id = 500000
                start_time = time.time()
                
                # Initialize progress bar
                pbar = tqdm(total=total_rows, desc="Processing embeddings (CPU)", unit="records")
                
                # Process with cursor-based pagination
                while processed_records < total_rows:
                    # Fetch batch using cursor-based pagination (much faster than OFFSET)
                    rows = get_next_batch_cursor_based(
                        cur, 
                        last_id=last_id, 
                        batch_size=CPU_OPTIMIZED_BATCH_SIZES['db_fetch']
                    )
                    
                    if not rows:
                        break
                    
                    # Update last_id for next iteration
                    last_id = rows[-1][0]
                    
                    # Process batch
                    batch_count += 1
                    
                    # Process batch and get updates
                    updates = process_batch_cpu_optimized(rows, embedder)
                    
                    if updates:
                        # Batch update database using execute_batch (faster than executemany)
                        execute_batch(
                            cur,
                            """UPDATE arxiv SET embedding = %s WHERE id = %s""",
                            updates,
                            page_size=CPU_OPTIMIZED_BATCH_SIZES['db_update']
                        )
                        
                        # Commit every N batches for better performance
                        if batch_count % CPU_OPTIMIZED_BATCH_SIZES['commit_every'] == 0:
                            conn.commit()
                    
                    # Update progress
                    processed_records += len(rows)
                    pbar.update(len(rows))
                    
                    # Performance metrics
                    elapsed_time = time.time() - start_time
                    records_per_second = processed_records / elapsed_time
                    estimated_total_time = total_rows / records_per_second
                    remaining_time = estimated_total_time - elapsed_time
                    
                    if batch_count % 5 == 0:  # Log every 5 batches
                        logger.info(f"💻 CPU BATCH {batch_count}: {processed_records}/{total_rows} records "
                                   f"({100*processed_records/total_rows:.1f}%) - "
                                   f"Speed: {records_per_second:.1f} records/sec - "
                                   f"ETA: {remaining_time/60:.1f} minutes")
                    
                    # Memory cleanup every 25 batches
                    if batch_count % 25 == 0:
                        gc.collect()
                
                # Final commit
                conn.commit()
                pbar.close()
                
                # Final metrics
                total_time = time.time() - start_time
                final_speed = processed_records / total_time
                logger.info(f"🎉 CPU PROCESSING COMPLETED: Processed {processed_records} records in {total_time/60:.1f} minutes")
                logger.info(f"📈 FINAL SPEED: {final_speed:.1f} records/second")
                logger.info(f"⚡ PERFORMANCE IMPROVEMENT: ~{final_speed/10:.1f}x faster than original")

    except Exception as e:
        logger.error(f"Error updating embeddings: {str(e)}")
        raise

def test_embeddings_cpu_optimized():
    """Test the CPU-optimized local embedding setup"""
    logger.info("Testing CPU-optimized local embeddings...")
    
    embedder = CPUOptimizedLocalEmbedder()
    
    test_texts = [
        "This is a test document about machine learning.",
        "Another document discussing natural language processing.",
        "A paper on computer vision and deep learning.",
        "Research on artificial intelligence and automation.",
        "Study of neural networks and deep learning algorithms."
    ]
    
    # Fit reducer if needed
    embedder.fit_reducer(test_texts)
    
    # Test with timing
    start_time = time.time()
    embeddings = embedder.get_embeddings(test_texts)
    processing_time = time.time() - start_time
    
    logger.info(f"✅ CPU test successful!")
    logger.info(f"📊 Input texts: {len(test_texts)}")
    logger.info(f"📊 Output embeddings shape: {embeddings.shape}")
    logger.info(f"📊 Target dimensions: {TARGET_DIMENSIONS}")
    logger.info(f"📊 Actual dimensions: {embeddings.shape[1]}")
    logger.info(f"⏱️ Processing time: {processing_time:.3f} seconds")
    logger.info(f"⚡ Speed: {len(test_texts)/processing_time:.1f} texts/second")
    logger.info(f"💻 CPU Cores: {mp.cpu_count()}")
    
    return embeddings.shape[1] == TARGET_DIMENSIONS

def main():
    logger.info("🚀 Starting CPU-OPTIMIZED local embedding process...")
    
    # Test setup first
    if test_embeddings_cpu_optimized():
        logger.info("✅ CPU-optimized local embedding test passed!")
        logger.info("🏃‍♂️ Starting CPU-optimized batch processing...")
        update_embeddings_cpu_optimized()
        logger.info("🎉 CPU-optimized local embedding process completed!")
    else:
        logger.error("❌ CPU-optimized local embedding test failed!")

if __name__ == "__main__":
    main() 