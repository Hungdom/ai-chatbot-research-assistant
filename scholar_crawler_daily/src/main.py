import argparse
import logging
from pathlib import Path
from crawler.scholar_crawler import ScholarCrawler
from config.settings import MATH_TOPICS, CRAWLER_CONFIG
from crawler.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Crawl academic papers from Semantic Scholar')
    parser.add_argument('--year', type=int, help='Year to filter papers')
    parser.add_argument('--topic', type=str, help='Math topic to filter papers')
    parser.add_argument('--max-results', type=int, help='Maximum number of results to fetch')
    args = parser.parse_args()

    # Initialize crawler
    crawler = ScholarCrawler()
    
    # Set up filters
    year = args.year or CRAWLER_CONFIG['year_range']['start']
    topic = args.topic
    max_results = args.max_results

    # Crawl papers
    try:
        stored_count = crawler.crawl_math_papers(
            topic=topic,
            year=year,
            max_results=max_results
        )
        logger.info(f"Successfully processed {stored_count} papers")
    except Exception as e:
        logger.error(f"Error during crawling: {e}")
        raise

if __name__ == "__main__":
    main() 