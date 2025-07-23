import time
import random
import json
import csv
from datetime import datetime
from pathlib import Path
from semanticscholar import SemanticScholar
from typing import List, Dict, Optional
from database.session import get_session, init_db
from database.models import Paper, Base
from config.settings import CRAWLER_CONFIG, MATH_TOPICS
from crawler.utils.logger import setup_logger

logger = setup_logger(__name__)

class ScholarCrawler:
    def __init__(self):
        logger.info("Initializing crawler and database connection...")
        try:
            # Initialize database tables
            init_db()
            self.session = get_session()
            self.sch = SemanticScholar()
            # Create data directory if it doesn't exist
            self.data_dir = Path(__file__).parent / 'data'
            self.data_dir.mkdir(exist_ok=True)
            logger.info("Crawler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize crawler: {e}")
            raise

    def __del__(self):
        """Clean up the session when the crawler is destroyed"""
        if hasattr(self, 'session'):
            logger.info("Closing database session")
            try:
                self.session.close()
            except Exception as e:
                logger.error(f"Error closing session: {e}")

    def _random_delay(self):
        """Add a random delay between requests"""
        delay = CRAWLER_CONFIG['delay_between_requests'] + random.uniform(1, 3)
        logger.debug(f"Waiting {delay:.2f} seconds before next request")
        time.sleep(delay)

    def _build_search_query(self, topic: Optional[str] = None, year: Optional[int] = None, custom_query: Optional[str] = None) -> str:
        """Build a search query based on filters"""
        query_parts = []
        
        if custom_query:
            query_parts.append(custom_query)
        elif topic:
            query_parts.append(topic)
        else:
            query_parts.append("mathematics")
            
        if year:
            query_parts.append(f"year:{year}")
            
        return " ".join(query_parts)

    def search_papers(self, topic: Optional[str] = None, year: Optional[int] = None, year_end: Optional[int] = None, custom_query: Optional[str] = None, max_results: Optional[int] = None, max_retries: int = 3) -> List[Dict]:
        """Search for papers using Semantic Scholar API"""
        query = self._build_search_query(topic, year, custom_query)
        logger.info(f"Searching for papers with query: '{query}'")
        
        processed_papers = []
        total_processed = 0
        limit = 100  # Maximum allowed by Semantic Scholar API per request
        
        # Create CSV file at the start
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_str = f"_{topic}" if topic else ""
        year_str = f"_{year}" if year else ""
        year_end_str = f"_{year_end}" if year_end else ""
        csv_filename = self.data_dir / f"papers{topic_str}{year_str}{year_end_str}_{timestamp}.csv"
        
        # Define CSV headers
        fieldnames = ['title', 'authors', 'abstract', 'year', 'citations', 'url', 'fields_of_study']
        
        # Create CSV file with headers
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        # First, get total count of papers
        try:
            # Initial search to get total count
            initial_results = self.sch.search_paper(
                query,
                limit=1,  # Just get one result to check if there are any papers
                fields=['title', 'authors', 'abstract', 'year', 
                       'citationCount', 'url', 'fieldsOfStudy']
            )
            
            if not initial_results:
                logger.info("No papers found for the query")
                return processed_papers
                
            # Now get all papers in batches
            while True:
                for attempt in range(max_retries):
                    try:
                        # Search papers with current limit
                        papers = self.sch.search_paper(
                            query, 
                            limit=limit,
                            fields=['title', 'authors', 'abstract', 'year', 
                                   'citationCount', 'url', 'fieldsOfStudy']
                        )
                        
                        if not papers:
                            logger.info("No more papers found")
                            return processed_papers
                        
                        for paper in papers:
                            try:
                                # Extract author names from the paper object
                                authors = ', '.join([author.name for author in paper.authors]) if paper.authors else ''
                                
                                # Get fields of study
                                fields = paper.fieldsOfStudy if hasattr(paper, 'fieldsOfStudy') else []
                                
                                paper_data = {
                                    'title': paper.title or '',
                                    'authors': authors,
                                    'abstract': paper.abstract or '',
                                    'year': paper.year,
                                    'citations': paper.citationCount or 0,
                                    'url': paper.url or '',
                                    'fields_of_study': fields
                                }
                                
                                # Only process papers that have at least a title
                                if paper_data['title']:
                                    # Store in database
                                    if self.store_paper(paper_data):
                                        logger.info(f"Stored in database: {paper_data['title']}")
                                    
                                    # Append to CSV
                                    try:
                                        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                            paper_row = {
                                                'title': paper_data['title'],
                                                'authors': paper_data['authors'],
                                                'abstract': paper_data['abstract'].replace('\n', ' ').replace('\r', ' '),
                                                'year': paper_data['year'],
                                                'citations': paper_data['citations'],
                                                'url': paper_data['url'],
                                                'fields_of_study': ','.join(paper_data['fields_of_study'])
                                            }
                                            writer.writerow(paper_row)
                                        logger.info(f"Appended to CSV: {paper_data['title']}")
                                    except Exception as e:
                                        logger.error(f"Error writing to CSV: {e}")
                                    
                                    processed_papers.append(paper_data)
                                    total_processed += 1
                                    
                                    # Log progress every 5 papers
                                    if total_processed % 5 == 0:
                                        logger.info(f"Progress: {total_processed} papers processed")
                                
                                # Add small delay between processing papers
                                self._random_delay()
                            except Exception as e:
                                logger.warning(f"Error processing paper: {e}")
                                continue
                        
                        # If we got fewer papers than the limit, we've reached the end
                        if len(papers) < limit:
                            logger.info(f"Reached end of results. Total papers processed: {total_processed}")
                            return processed_papers
                        
                        # If max_results is set and we've reached it, stop
                        if max_results and total_processed >= max_results:
                            logger.info(f"Reached maximum number of papers to process ({max_results})")
                            return processed_papers[:max_results]
                        
                        break  # Break retry loop on success
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                            self._random_delay()
                        else:
                            logger.error(f"All attempts failed: {e}")
                            return processed_papers
                
                # Add delay between batches
                self._random_delay()
                
                # Increase limit for next batch
                limit = min(limit * 2, 100)  # Double the limit but don't exceed 100
                
        except Exception as e:
            logger.error(f"Error in initial search: {e}")
            return processed_papers
        
        return processed_papers

    def store_paper(self, paper_data: Dict) -> bool:
        """Store a single paper in the database"""
        try:
            logger.info(f"Attempting to store paper: {paper_data['title']}")
            
            # Check if paper already exists
            existing_paper = self.session.query(Paper).filter_by(title=paper_data['title']).first()
            if existing_paper:
                logger.info(f"Paper already exists: {paper_data['title']}")
                return True

            # Create new paper
            paper = Paper(
                title=paper_data['title'],
                authors=paper_data['authors'],
                abstract=paper_data['abstract'],
                year=paper_data['year'],
                citations=paper_data['citations'],
                url=paper_data['url']
            )
            
            self.session.add(paper)
            try:
                self.session.commit()
                logger.info(f"Successfully committed paper to database: {paper_data['title']}")
                return True
            except Exception as e:
                logger.error(f"Error committing paper {paper_data['title']}: {e}")
                self.session.rollback()
                return False
        except Exception as e:
            logger.error(f"Error storing paper {paper_data['title']}: {e}")
            self.session.rollback()
            return False

    def save_crawled_data(self, papers: List[Dict], stored_count: int):
        """Save crawled data to a JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.data_dir / f"crawled_papers_{timestamp}.json"
            
            data = {
                "timestamp": timestamp,
                "total_papers_found": len(papers),
                "total_papers_stored": stored_count,
                "papers": papers
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved crawled data to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving crawled data: {e}")
            return None

    def save_to_csv(self, papers: List[Dict], topic: Optional[str] = None, year: Optional[int] = None):
        """Save papers to a CSV file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_str = f"_{topic}" if topic else ""
            year_str = f"_{year}" if year else ""
            filename = self.data_dir / f"papers{topic_str}{year_str}_{timestamp}.csv"
            
            # Define CSV headers
            fieldnames = ['title', 'authors', 'abstract', 'year', 'citations', 'url', 'fields_of_study']
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for paper in papers:
                    # Clean the data for CSV writing
                    paper_row = {
                        'title': paper['title'],
                        'authors': paper['authors'],
                        'abstract': paper['abstract'].replace('\n', ' ').replace('\r', ' '),
                        'year': paper['year'],
                        'citations': paper['citations'],
                        'url': paper['url'],
                        'fields_of_study': ','.join(paper.get('fields_of_study', []))
                    }
                    writer.writerow(paper_row)
            
            logger.info(f"Successfully saved {len(papers)} papers to CSV: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving papers to CSV: {e}")
            return None

    def crawl_math_papers(self, topic: Optional[str] = None, year: Optional[int] = None, year_end: Optional[int] = None, custom_query: Optional[str] = None, max_results: Optional[int] = None):
        """Main function to crawl math papers with filters"""
        try:
            logger.info("Starting paper crawl...")
            
            # Validate topic if provided
            if topic and topic not in MATH_TOPICS:
                logger.warning(f"Invalid topic: {topic}. Using default search.")
                topic = None
                
            # Validate year if provided
            if year:
                year_range = CRAWLER_CONFIG['year_range']
                if not (year_range['start'] <= year <= year_range['end']):
                    logger.warning(f"Year {year} out of range. Using default year range.")
                    year = None
            
            papers = self.search_papers(topic=topic, year=year, year_end=year_end, custom_query=custom_query, max_results=max_results)
            
            # Save crawled data to JSON as backup
            self.save_crawled_data(papers, len(papers))
            
            logger.info(f"Paper crawl completed successfully. Processed {len(papers)} papers.")
            return len(papers)
        except Exception as e:
            logger.error(f"Error during paper crawl: {e}")
            raise
        finally:
            # Ensure session is closed
            logger.info("Closing database session")
            try:
                self.session.close()
            except Exception as e:
                logger.error(f"Error closing session: {e}")