#!/usr/bin/env python3
"""
Database optimization script for free tier EC2
Adds indexes to improve query performance
"""

import asyncio
import logging
from database import get_db
from sqlalchemy import text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_database():
    """
    Add indexes to improve database performance on free tier EC2
    """
    logger.info("Starting database optimization...")
    
    try:
        with get_db() as db:
            # Enable pg_trgm extension for text search (if not already enabled)
            try:
                db.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
                logger.info("✓ pg_trgm extension enabled")
            except Exception as e:
                logger.warning(f"Could not enable pg_trgm: {e}")
            
            # Create indexes for better performance
            indexes = [
                # Index for published_date DESC (most common sorting)
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_published_date_desc ON arxiv (published_date DESC);",
                
                # Index for primary_category + published_date (common filter + sort)
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_primary_category_date ON arxiv (primary_category, published_date DESC);",
                
                # GIN index for title text search
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_title_gin ON arxiv USING gin (title gin_trgm_ops);",
                
                # GIN index for categories array
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_categories_gin ON arxiv USING gin (categories);",
                
                # Index for arxiv_id (should already exist but make sure)
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_arxiv_id ON arxiv (arxiv_id);",
                
                # Index for embedding existence check
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embedding_exists ON arxiv (embedding) WHERE embedding IS NOT NULL;",
            ]
            
            for index_sql in indexes:
                try:
                    logger.info(f"Creating index: {index_sql.split('idx_')[1].split(' ')[0]}")
                    db.execute(text(index_sql))
                    logger.info("✓ Index created successfully")
                except Exception as e:
                    logger.warning(f"Index creation failed: {e}")
            
            # Commit all changes
            db.commit()
            logger.info("✓ Database optimization completed")
            
            # Analyze tables to update statistics
            db.execute(text("ANALYZE arxiv;"))
            db.commit()
            logger.info("✓ Table statistics updated")
            
    except Exception as e:
        logger.error(f"Database optimization failed: {e}")
        raise

def check_database_performance():
    """
    Check database performance and provide recommendations
    """
    logger.info("Checking database performance...")
    
    try:
        with get_db() as db:
            # Check table size
            result = db.execute(text("""
                SELECT 
                    COUNT(*) as total_papers,
                    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as papers_with_embeddings
                FROM arxiv
            """)).fetchone()
            
            total_papers = result[0]
            papers_with_embeddings = result[1]
            
            logger.info(f"📊 Database stats:")
            logger.info(f"   Total papers: {total_papers:,}")
            logger.info(f"   Papers with embeddings: {papers_with_embeddings:,}")
            
            # Check indexes
            result = db.execute(text("""
                SELECT indexname, tablename 
                FROM pg_indexes 
                WHERE tablename = 'arxiv'
                ORDER BY indexname
            """)).fetchall()
            
            logger.info(f"📈 Indexes on arxiv table:")
            for index in result:
                logger.info(f"   - {index[0]}")
            
            # Performance recommendations
            logger.info("\n🚀 Performance recommendations:")
            
            if total_papers > 100000:
                logger.info("   ✓ Large dataset - indexes are crucial")
            
            if papers_with_embeddings == 0:
                logger.info("   ⚠️  No embeddings found - semantic search will be limited")
                logger.info("   💡 Consider generating embeddings for better search results")
            
            # Check for slow queries
            logger.info("   💡 For free tier EC2:")
            logger.info("   - Use category-based searches when possible")
            logger.info("   - Limit result sets to 10-20 papers")
            logger.info("   - Avoid complex text searches on large datasets")
            logger.info("   - Use year filters to reduce search scope")
            
    except Exception as e:
        logger.error(f"Performance check failed: {e}")

if __name__ == "__main__":
    print("🔧 Database Optimization for Free Tier EC2")
    print("=" * 50)
    
    # Run optimization
    optimize_database()
    
    # Check performance
    check_database_performance()
    
    print("\n✅ Database optimization complete!")
    print("📈 Your research assistant should now be much faster!") 