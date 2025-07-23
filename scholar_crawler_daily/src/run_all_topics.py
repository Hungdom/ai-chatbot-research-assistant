import os
import time
from crawler.scholar_crawler import ScholarCrawler
from config.settings import MATH_TOPICS
from crawler.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    # Get configuration from environment variables
    year = int(os.getenv('YEAR_START', '2000'))
    year_end = int(os.getenv('YEAR_END', '2000'))
    max_results = int(os.getenv('MAX_RESULTS', '1000')) if os.getenv('MAX_RESULTS') else None
    
    logger.info(f"Starting crawler for year {year} to {year_end} with max_results={max_results}")
    
    # Initialize crawler
    crawler = ScholarCrawler()
    
    total_papers = 0
    for topic in MATH_TOPICS:
        try:
            logger.info(f"Processing topic: {topic}")
            papers_count = crawler.crawl_math_papers(
                topic=topic,
                year=year,
                max_results=max_results
            )
            total_papers += papers_count
            logger.info(f"Completed topic {topic}. Found {papers_count} papers")
            
            # Add delay between topics
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error processing topic {topic}: {e}")
            continue
    
    logger.info(f"Crawling completed. Total papers processed: {total_papers}")

if __name__ == "__main__":
    main() 