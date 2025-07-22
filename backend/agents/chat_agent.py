from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
import openai
from datetime import datetime
from config.prompts import CHAT_AGENT_PROMPTS
from database import get_db, Arxiv, ChatSession
from sqlalchemy import text, func, desc, or_, extract, and_
from sqlalchemy.sql import select
import numpy as np
import json
import logging
import re
from sqlalchemy.sql import bindparam
import uuid

class ChatAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        # Set up detailed logging
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user input and generate a response
        """
        try:
            self.logger.info("=== Starting Chat Processing ===")
            self.logger.info(f"Input data: {json.dumps(input_data, indent=2)}")
            
            query = input_data.get("query", "")
            context = input_data.get("context", {})
            session_id = input_data.get("session_id")
            
            # Get or create chat session
            with get_db() as db:
                if session_id:
                    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
                else:
                    session = None
                
                if not session:
                    session = ChatSession(
                        session_id=str(uuid.uuid4()),
                        messages=[],
                        context={}
                    )
                    db.add(session)
                    db.commit()
                    session_id = session.session_id
                
                # Add user message to session
                session.messages.append({
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.utcnow().isoformat()
                })
                db.commit()
            
            # Analyze the query to determine intent
            self.logger.info("Analyzing query intent...")
            intent = await self._analyze_intent(query)
            self.logger.info(f"Intent analysis result: {json.dumps(intent, indent=2)}")
            
            # Search for relevant papers using both semantic and metadata search
            arxiv_results = []
            if intent.get("needs_search", False):
                self.logger.info("Searching for relevant papers...")
                arxiv_results = await self._search_relevant_papers(query, intent)
                self.logger.info(f"Found {len(arxiv_results)} relevant papers")
                
                # Get additional insights if we found papers
                if arxiv_results:
                    self.logger.info("Generating research insights...")
                    insights = await self._generate_research_insights(arxiv_results, intent)
                    intent["insights"] = insights
                    self.logger.info(f"Generated insights: {json.dumps(insights, indent=2)}")
            
            # Generate response based on intent and search results
            self.logger.info("Generating response...")
            response_data = await self._generate_response(query, intent, arxiv_results, context)
            self.logger.info(f"Generated response: {response_data.get('response', '')[:200]}...")  # Log first 200 chars
            
            # Add assistant response to session
            with get_db() as db:
                session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
                if session:
                    session.messages.append({
                        "role": "assistant",
                        "content": response_data.get("response", ""),
                        "html_content": response_data.get("html_response", ""),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    db.commit()
            
            self.logger.info("=== Chat Processing Complete ===")
            
            # Ensure we always return a properly structured response
            return {
                "response": response_data.get("response", ""),
                "html_response": response_data.get("html_response", ""),
                "arxiv": arxiv_results,
                "intent": intent,
                "session_id": session_id
            }
        except Exception as e:
            self.logger.error(f"Error in chat processing: {str(e)}", exc_info=True)
            # Return a safe error response
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "html_response": "<div class='error-message'>I apologize, but I encountered an error processing your request. Please try again.</div>",
                "arxiv": [],
                "intent": {"type": "error", "needs_search": False, "search_params": {}, "follow_up_questions": []},
                "session_id": session_id if 'session_id' in locals() else None
            }
    
    async def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user's query to determine intent and required actions
        """
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": CHAT_AGENT_PROMPTS["intent_analysis"]},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        
        # Parse the response to determine intent
        intent_text = completion.choices[0].message.content
        intent = {
            "type": "general_query",  # default type
            "needs_search": False,
            "search_params": {},
            "follow_up_questions": []
        }
        
        # Determine if we need to search for papers
        if any(keyword in query.lower() for keyword in ["paper", "research", "study", "find", "search", "look for"]):
            intent["needs_search"] = True
            intent["type"] = "paper_search"
            
            # Extract search parameters
            if "year" in query.lower():
                intent["search_params"]["year"] = self._extract_year(query)
            if "category" in query.lower() or "field" in query.lower():
                intent["search_params"]["category"] = self._extract_category(query)
        
        # Generate follow-up questions
        intent["follow_up_questions"] = await self._generate_follow_up_questions(query, intent)
        
        return intent
    
    async def _search_relevant_papers(self, query: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Optimized paper search with efficient pre-filtering and minimal API calls
        """
        try:
            self.logger.info("Starting optimized paper search...")
            
            with get_db() as db:
                # 1. FAST PRE-FILTERING
                # Extract key terms
                query_lower = query.lower().strip()
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
                key_terms = [word for word in re.findall(r'\b\w+\b', query_lower) if word not in stop_words and len(word) > 2]
                
                # 2. BUILD EFFICIENT QUERY
                base_query = select(Arxiv)
                conditions = []
                
                # Title/Abstract filtering with key terms
                if key_terms:
                    text_conditions = []
                    for term in key_terms[:3]:  # Top 3 terms only
                        text_conditions.extend([
                            Arxiv.title.ilike(f'%{term}%'),
                            Arxiv.abstract.ilike(f'%{term}%')
                        ])
                    
                    if text_conditions:
                        conditions.append(or_(*text_conditions))
                
                # Category filtering
                search_params = intent.get("search_params", {})
                if search_params.get("category"):
                    category = search_params["category"]
                    conditions.append(
                        or_(
                            Arxiv.categories.contains([category]),
                            Arxiv.primary_category == category
                        )
                    )
                
                # Year filtering
                if search_params.get("year"):
                    year = search_params["year"]
                    conditions.append(extract('year', Arxiv.published_date) == year)
                
                # Apply conditions
                if conditions:
                    base_query = base_query.where(or_(*conditions))
                
                # 3. TRY EMBEDDING SEARCH FIRST (if available)
                papers = []
                
                # Quick embedding check - don't generate embedding if no papers have embeddings
                has_embeddings = db.execute(
                    text("SELECT EXISTS(SELECT 1 FROM arxiv WHERE embedding IS NOT NULL LIMIT 1)")
                ).scalar()
                
                if has_embeddings:
                    query_embedding = await self._get_embedding(query)
                    if query_embedding:
                        try:
                            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                            similarity_expr = text(
                                f"cosine_similarity(arxiv.embedding, '{embedding_str}'::vector) as similarity"
                            )
                            
                            # Embedding query with pre-filtering
                            embedding_query = select(Arxiv, similarity_expr).select_from(Arxiv).where(
                                and_(
                                    Arxiv.embedding.is_not(None),
                                    *conditions if conditions else [text('true')]
                                )
                            ).order_by(text("similarity DESC")).limit(12)
                            
                            results = db.execute(embedding_query).all()
                            
                            for result in results:
                                if isinstance(result, tuple) and len(result) == 2:
                                    paper, similarity = result
                                    if similarity and similarity > 0.2:  # Higher threshold
                                        papers.append({
                                            "arxiv_id": paper.arxiv_id,
                                            "title": paper.title,
                                            "abstract": paper.abstract,
                                            "authors": paper.authors,
                                            "categories": paper.categories,
                                            "published_date": paper.published_date.isoformat() if paper.published_date else None,
                                            "doi": paper.doi,
                                            "primary_category": paper.primary_category,
                                            "similarity": float(similarity)
                                        })
                            
                            if papers:
                                self.logger.info(f"Found {len(papers)} papers using embedding search")
                                return papers
                        except Exception as e:
                            self.logger.warning(f"Embedding search failed: {str(e)}")
                
                # 4. FALLBACK TO KEYWORD SEARCH
                # Order by recent papers first, limit to reduce processing
                base_query = base_query.order_by(Arxiv.published_date.desc()).limit(25)
                results = db.execute(base_query).scalars().all()
                
                # Calculate similarity and filter
                papers_with_scores = []
                for paper in results:
                    similarity = self._calculate_keyword_similarity(query, paper)
                    if similarity > 0.0:
                        papers_with_scores.append((paper, similarity))
                
                # Sort by similarity and format
                papers_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                papers = []
                for paper, similarity in papers_with_scores[:12]:  # Top 12 results
                    papers.append({
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "abstract": paper.abstract,
                        "authors": paper.authors,
                        "categories": paper.categories,
                        "published_date": paper.published_date.isoformat() if paper.published_date else None,
                        "doi": paper.doi,
                        "primary_category": paper.primary_category,
                        "similarity": float(similarity)
                    })
                
                self.logger.info(f"Found {len(papers)} papers using keyword search")
                return papers
                
        except Exception as e:
            self.logger.error(f"Error in paper search: {str(e)}", exc_info=True)
            return []

    async def _search_papers_by_keywords(self, query: str, intent: Dict[str, Any], db) -> List[Dict[str, Any]]:
        """
        Fallback search using keyword matching when embeddings are not available
        """
        try:
            self.logger.info("Performing keyword-based search...")
            
            # Extract keywords from query
            keywords = query.lower().split()
            
            # Build keyword search query
            keyword_query = select(Arxiv)
            
            # Add keyword conditions
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append(
                    or_(
                        Arxiv.title.ilike(f"%{keyword}%"),
                        Arxiv.abstract.ilike(f"%{keyword}%")
                    )
                )
            
            if keyword_conditions:
                keyword_query = keyword_query.where(or_(*keyword_conditions))
            
            # Add filters based on search parameters
            if intent.get("search_params", {}).get("year"):
                year = intent["search_params"]["year"]
                keyword_query = keyword_query.where(
                    extract('year', Arxiv.published_date) == year
                )

            if intent.get("search_params", {}).get("category"):
                category = intent["search_params"]["category"]
                keyword_query = keyword_query.where(
                    Arxiv.categories.contains([category])
                )
            
            # Order by date and limit
            keyword_query = keyword_query.order_by(Arxiv.published_date.desc()).limit(10)
            
            # Execute query
            results = db.execute(keyword_query).scalars().all()
            
            # Convert results to dictionaries
            papers = []
            for paper in results:
                # Calculate keyword-based similarity
                similarity = self._calculate_keyword_similarity(query, paper)
                
                paper_dict = {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "categories": paper.categories,
                    "published_date": paper.published_date.isoformat() if paper.published_date else None,
                    "doi": paper.doi,
                    "primary_category": paper.primary_category,
                    "similarity": similarity
                }
                papers.append(paper_dict)
            
            # Sort by similarity score (descending)
            papers.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)

            self.logger.info(f"Found {len(papers)} papers using keyword search")
            return papers
            
        except Exception as e:
            self.logger.error(f"Error in keyword search: {str(e)}", exc_info=True)
            return []
    
    async def _get_related_papers(self, papers: List[Arxiv], db) -> List[Arxiv]:
        """
        Find related papers based on categories, authors, and citations
        """
        try:
            self.logger.info("Starting related papers search")
            
            # Extract categories and authors
            categories = set()
            authors = set()
            for paper in papers:
                categories.update(paper.categories)
                authors.update(paper.authors)
            
            self.logger.info(f"Found {len(categories)} unique categories and {len(authors)} unique authors")
            
            # Build a more efficient query using EXISTS
            related_query = select(Arxiv).where(
                Arxiv.id.notin_([p.id for p in papers])
            )
            
            # Use EXISTS for better performance with arrays
            category_conditions = []
            for category in categories:
                category_conditions.append(
                    text("EXISTS (SELECT 1 FROM unnest(categories) cat WHERE cat = :category)")
                    .bindparams(category=category)
                )
            
            author_conditions = []
            for author in authors:
                author_conditions.append(
                    text("EXISTS (SELECT 1 FROM unnest(authors) auth WHERE auth = :author)")
                    .bindparams(author=author)
                )
            
            if category_conditions or author_conditions:
                related_query = related_query.where(or_(*category_conditions, *author_conditions))
            
            # Execute query with limit
            self.logger.info("Executing related papers query...")
            results = db.execute(
                related_query.order_by(Arxiv.published_date.desc())
                .limit(5)
            ).scalars().all()
            
            self.logger.info(f"Found {len(results)} related papers")
            return results
        except Exception as e:
            self.logger.error(f"Error finding related papers: {str(e)}", exc_info=True)
            return []
    
    async def _generate_research_insights(self, papers: List[Dict[str, Any]], intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights about the research papers
        """
        try:
            self.logger.info("Starting research insights generation")
            
            # Analyze trends
            self.logger.info("Analyzing trends...")
            trends = self._analyze_trends(papers)
            self.logger.info(f"Trend analysis complete: {json.dumps(trends, indent=2)}")
            
            # Identify key authors and institutions
            self.logger.info("Analyzing authors and institutions...")
            authors_analysis = self._analyze_authors(papers)
            self.logger.info(f"Author analysis complete: {json.dumps(authors_analysis, indent=2)}")
            
            # Analyze methodology patterns
            self.logger.info("Analyzing methodologies...")
            methodology = self._analyze_methodology(papers)
            self.logger.info(f"Methodology analysis complete: {json.dumps(methodology, indent=2)}")
            
            # Identify research gaps
            self.logger.info("Identifying research gaps...")
            gaps = self._identify_research_gaps(papers)
            self.logger.info(f"Research gaps identified: {json.dumps(gaps, indent=2)}")
            
            insights = {
                "trends": trends,
                "authors": authors_analysis,
                "methodology": methodology,
                "gaps": gaps
            }
            
            self.logger.info("Research insights generation complete")
            return insights
        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}", exc_info=True)
            return {}
    
    def _analyze_trends(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze research trends from the papers
        """
        trends = {
            "temporal": {},
            "topics": {},
            "categories": {}
        }
        
        # Analyze temporal trends
        for paper in papers:
            year = paper["published_date"][:4]
            trends["temporal"][year] = trends["temporal"].get(year, 0) + 1
        
        # Analyze topic trends
        for paper in papers:
            # Extract key topics from title and abstract
            topics = self._extract_topics(paper["title"] + " " + paper["abstract"])
            for topic in topics:
                trends["topics"][topic] = trends["topics"].get(topic, 0) + 1
        
        # Analyze category distribution
        for paper in papers:
            for category in paper["categories"]:
                trends["categories"][category] = trends["categories"].get(category, 0) + 1
        
        return trends
    
    def _analyze_authors(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze author patterns and collaborations
        """
        authors = {}
        institutions = {}
        
        for paper in papers:
            for author in paper["authors"]:
                # Count author frequency
                authors[author] = authors.get(author, 0) + 1
                
                # TODO: Add institution analysis if available
                # if "institution" in author:
                #     institutions[author["institution"]] = institutions.get(author["institution"], 0) + 1
        
        return {
            "top_authors": dict(sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_institutions": dict(sorted(institutions.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def _analyze_methodology(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze research methodologies used in the papers
        """
        methodologies = {}
        
        # Common methodology keywords
        method_keywords = {
            "experimental": ["experiment", "trial", "study", "test"],
            "theoretical": ["theory", "theoretical", "model", "framework"],
            "review": ["review", "survey", "meta-analysis"],
            "case_study": ["case study", "case analysis"],
            "simulation": ["simulation", "simulate", "modeling"]
        }
        
        for paper in papers:
            text = paper["title"] + " " + paper["abstract"]
            for method, keywords in method_keywords.items():
                if any(keyword in text.lower() for keyword in keywords):
                    methodologies[method] = methodologies.get(method, 0) + 1
        
        return methodologies
    
    def _identify_research_gaps(self, papers: List[Dict[str, Any]]) -> List[str]:
        """
        Identify potential research gaps based on paper analysis
        """
        gaps = []
        
        # Analyze temporal gaps
        years = sorted([int(p["published_date"][:4]) for p in papers])
        if years:
            year_gaps = [years[i+1] - years[i] for i in range(len(years)-1)]
            if year_gaps and max(year_gaps) > 2:
                gaps.append(f"Significant time gap in research between {years[year_gaps.index(max(year_gaps))]} and {years[year_gaps.index(max(year_gaps))+1]}")
        
        # Analyze methodological gaps
        methodologies = self._analyze_methodology(papers)
        underrepresented = [m for m, count in methodologies.items() if count < len(papers) * 0.2]
        if underrepresented:
            gaps.append(f"Limited research using {', '.join(underrepresented)} methodologies")
        
        return gaps
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract key topics from text
        """
        # TODO: Implement more sophisticated topic extraction
        # For now, use simple keyword matching
        common_topics = [
            "machine learning", "deep learning", "neural networks",
            "artificial intelligence", "data science", "big data",
            "computer vision", "natural language processing",
            "reinforcement learning", "optimization"
        ]
        
        return [topic for topic in common_topics if topic in text.lower()]
    
    def _calculate_keyword_similarity(self, query: str, paper: Arxiv) -> float:
        """
        Fast and smart similarity calculation optimized for limited database context
        Uses weighted scoring based on field importance and relevance
        """
        try:
            query_lower = query.lower().strip()
            if not query_lower:
                return 0.0
            
            # Extract key terms from query (remove common words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
            query_words = [word for word in re.findall(r'\b\w+\b', query_lower) if word not in stop_words and len(word) > 2]
            
            if not query_words:
                return 0.0
            
            # Initialize scoring components
            title_score = 0.0
            abstract_score = 0.0
            category_score = 0.0
            author_score = 0.0
            
            # 1. TITLE SCORING (40% weight - most important)
            title_lower = paper.title.lower()
            title_words = re.findall(r'\b\w+\b', title_lower)
            
            # Exact phrase matching in title (high value)
            if query_lower in title_lower:
                title_score += 1.0
            
            # Word matching in title
            title_matches = sum(1 for word in query_words if word in title_lower)
            title_score += (title_matches / len(query_words)) * 0.8
            
            # Consecutive word matching (phrases)
            for i in range(len(query_words) - 1):
                phrase = ' '.join(query_words[i:i+2])
                if phrase in title_lower:
                    title_score += 0.3
            
            # 2. ABSTRACT SCORING (30% weight)
            abstract_lower = paper.abstract.lower()
            
            # Key word density in abstract
            abstract_matches = sum(1 for word in query_words if word in abstract_lower)
            abstract_score = (abstract_matches / len(query_words)) * 0.7
            
            # Boost for early occurrence in abstract (first 200 chars)
            abstract_start = abstract_lower[:200]
            early_matches = sum(1 for word in query_words if word in abstract_start)
            abstract_score += (early_matches / len(query_words)) * 0.3
            
            # 3. CATEGORY SCORING (20% weight)
            for category in paper.categories:
                category_lower = category.lower()
                
                # Direct category match
                if any(word in category_lower for word in query_words):
                    category_score += 0.5
                
                # Category hierarchy matching
                if category_lower.startswith('cs.') and any(tech_word in query_lower for tech_word in ['machine', 'learning', 'ai', 'algorithm', 'computer', 'data']):
                    category_score += 0.3
                elif category_lower.startswith('math.') and any(math_word in query_lower for math_word in ['equation', 'theorem', 'proof', 'analysis', 'algebra']):
                    category_score += 0.3
                elif category_lower.startswith('physics.') and any(phys_word in query_lower for phys_word in ['quantum', 'particle', 'energy', 'field']):
                    category_score += 0.3
            
            # 4. AUTHOR SCORING (10% weight)
            if paper.authors:
                for author in paper.authors:
                    author_lower = author.lower()
                    if any(word in author_lower for word in query_words):
                        author_score += 0.5
                        break  # Only count once per paper
            
            # 5. RECENCY BOOST (time-based relevance)
            recency_score = 0.0
            if paper.published_date:
                from datetime import datetime
                days_old = (datetime.now() - paper.published_date).days
                if days_old < 365:  # Last year
                    recency_score = 0.2
                elif days_old < 365 * 3:  # Last 3 years
                    recency_score = 0.1
            
            # 6. COMBINE SCORES WITH WEIGHTS
            final_score = (
                title_score * 0.40 +      # Title is most important
                abstract_score * 0.30 +   # Abstract content
                category_score * 0.20 +   # Category relevance
                author_score * 0.10 +     # Author matching
                recency_score            # Recency boost
            )
            
            # Cap at 1.0 and ensure minimum threshold
            final_score = min(final_score, 1.0)
            
            # Apply minimum threshold - papers below 0.1 are likely irrelevant
            return max(final_score, 0.0) if final_score >= 0.1 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text using OpenAI API
        """
        try:
            if not text or not text.strip():
                self.logger.warning("Empty or None text provided for embedding")
                return None
                
            self.logger.info(f"Generating embedding for text: {text[:100]}...")
            
            # Clean the text - remove extra whitespace and ensure it's not too long
            cleaned_text = text.strip()
            if len(cleaned_text) > 8000:  # OpenAI has token limits
                cleaned_text = cleaned_text[:8000]
                self.logger.warning(f"Text truncated to 8000 characters for embedding")
            
            response = await openai.Embedding.acreate(
                input=cleaned_text,
                model="text-embedding-ada-002"
            )
            
            if response and response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                self.logger.info(f"Successfully generated embedding with {len(embedding)} dimensions")
                return embedding
            else:
                self.logger.error("Invalid response from OpenAI embedding API")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}", exc_info=True)
            return None
    
    def _format_paper(self, paper: Arxiv) -> Dict[str, Any]:
        """
        Format a paper object for the response
        """
        return {
            "id": paper.id,
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "categories": paper.categories,
            "published_date": paper.published_date.isoformat() if paper.published_date else None,
            "updated_date": paper.updated_date.isoformat() if paper.updated_date else None,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
            "primary_category": paper.primary_category,
            "comment": paper.comment
        }
    
    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation: 1 token ≈ 4 characters)
        """
        return len(text) // 4

    def _truncate_papers_for_context(self, papers: List[Dict[str, Any]], max_tokens: int = 8000) -> List[Dict[str, Any]]:
        """
        Truncate papers list to fit within token limits
        """
        if not papers:
            return papers
        
        truncated_papers = []
        current_tokens = 0
        
        for paper in papers:
            # Create a minimal version of the paper for token counting
            paper_summary = {
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "abstract": paper["abstract"][:300] + "..." if len(paper["abstract"]) > 300 else paper["abstract"],
                "authors": paper["authors"][:3],  # Only first 3 authors
                "categories": paper["categories"],
                "published_date": paper["published_date"],
                "similarity": paper.get("similarity", 0.0)
            }
            
            # Estimate tokens for this paper
            paper_text = json.dumps(paper_summary)
            paper_tokens = self._count_tokens(paper_text)
            
            # Check if adding this paper would exceed the limit
            if current_tokens + paper_tokens > max_tokens:
                break
                
            truncated_papers.append(paper_summary)
            current_tokens += paper_tokens
        
        self.logger.info(f"Truncated papers: {len(papers)} -> {len(truncated_papers)} (estimated {current_tokens} tokens)")
        return truncated_papers

    async def _generate_response(self, query: str, intent: Dict[str, Any], arxiv_results: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a response based on the query, intent, and search results
        """
        try:
            # 1. TRUNCATE PAPERS TO PREVENT TOKEN OVERFLOW
            truncated_papers = self._truncate_papers_for_context(arxiv_results, max_tokens=6000)
            
            # 2. Prepare context for the AI with truncated data
            response_context = {
                "query": query,
                "intent_type": intent.get("type", "general_query"),
                "has_papers": len(truncated_papers) > 0,
                "papers": truncated_papers,
                "context": context,
                "search_params": intent.get("search_params", {}),
                "follow_up_questions": intent.get("follow_up_questions", [])
            }
            
            # 3. Select appropriate prompt template
            if intent.get("type") == "paper_search":
                if truncated_papers:
                    prompt = CHAT_AGENT_PROMPTS["paper_search_with_results"]
                else:
                    prompt = CHAT_AGENT_PROMPTS["paper_search_no_results"]
            else:
                prompt = CHAT_AGENT_PROMPTS["general_query"]
            
            # 4. Format the prompt with safe string formatting
            try:
                # Generate insights if we have papers and the prompt needs them
                insights_text = ""
                if truncated_papers and "{insights" in prompt:
                    insights = await self._generate_research_insights(truncated_papers, intent)
                    # Create CONCISE insights summary
                    insights_parts = []
                    if insights.get("trends", {}).get("categories"):
                        top_cats = list(insights["trends"]["categories"].keys())[:3]
                        insights_parts.append(f"- Top categories: {', '.join(top_cats)}")
                    if insights.get("authors", {}).get("top_authors"):
                        top_authors = list(insights["authors"]["top_authors"].keys())[:2]
                        insights_parts.append(f"- Key authors: {', '.join(top_authors)}")
                    insights_text = "\n".join(insights_parts)
                
                # Create COMPACT paper summary for prompt
                papers_summary = f"{len(truncated_papers)} papers found"
                if truncated_papers:
                    papers_summary += f" (top similarity: {truncated_papers[0].get('similarity', 0):.1%})"
                
                # Prepare all possible variables for the prompt
                format_variables = {
                    "query": response_context["query"],
                    "intent_type": response_context["intent_type"],
                    "has_papers": response_context["has_papers"],
                    "papers": papers_summary,  # Use summary instead of full data
                    "arxiv": papers_summary,   # Use summary instead of full data
                    "context": json.dumps(response_context["context"]) if response_context["context"] else "{}",
                    "search_params": json.dumps(response_context["search_params"]),
                    "follow_up_questions": json.dumps(response_context["follow_up_questions"]),
                    "insights": insights_text
                }
                
                formatted_prompt = prompt.format(**format_variables)
                
                # 5. FINAL TOKEN CHECK
                prompt_tokens = self._count_tokens(formatted_prompt)
                system_tokens = self._count_tokens(CHAT_AGENT_PROMPTS["system"])
                total_tokens = prompt_tokens + system_tokens
                
                self.logger.info(f"Token usage: {total_tokens} tokens (prompt: {prompt_tokens}, system: {system_tokens})")
                
                # If still too long, use minimal prompt
                if total_tokens > 15000:  # Leave buffer for response
                    self.logger.warning("Prompt still too long, using minimal format")
                    formatted_prompt = f"Query: {query}\nFound {len(truncated_papers)} relevant papers.\nProvide a concise response."
                
            except KeyError as e:
                self.logger.error(f"Error formatting prompt: {str(e)}")
                formatted_prompt = f"Query: {query}\nIntent: {intent.get('type', 'general_query')}\nResults: {len(truncated_papers)} papers found"
            
            # 6. Generate completion with smaller context
            completion = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": CHAT_AGENT_PROMPTS["system"]},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.7,
                max_tokens=1000  # Limit response length
            )
            
            response_text = completion.choices[0].message.content
            
            # 7. Generate HTML version using ORIGINAL paper list (not truncated)
            html_response = self._format_response_as_html(response_text, arxiv_results, intent)
            
            return {
                "response": response_text,
                "html_response": html_response,
                "arxiv": arxiv_results,  # Return full results for frontend
                "intent": intent
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "html_response": "<div class='error-message'>I apologize, but I encountered an error processing your request. Please try again.</div>",
                "arxiv": [],
                "intent": intent
            }
    
    def _format_response_as_html(self, response_text: str, papers: List[Dict[str, Any]], intent: Dict[str, Any]) -> str:
        """
        Format the response as HTML with enhanced styling for structured responses
        """
        import re
        
        html_parts = ['<div class="research-assistant-response">']
        
        # Convert structured markdown to HTML
        lines = response_text.split('\n')
        current_section = []
        in_list = False
        list_type = None
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_section:
                    html_parts.append('<br/>')
                continue
                
            # Headers
            if line.startswith('## '):
                if in_list:
                    html_parts.append(f'</{list_type}>')
                    in_list = False
                header_text = line[3:].strip()
                # Extract emoji and text
                emoji_match = re.match(r'^([^\w\s]+)\s*\*?\*?(.+?)\*?\*?$', header_text)
                if emoji_match:
                    emoji, title = emoji_match.groups()
                    html_parts.append(f'<div class="section-header"><span class="section-emoji">{emoji}</span><h2 class="section-title">{title.strip()}</h2></div>')
                else:
                    html_parts.append(f'<h2 class="section-title">{header_text}</h2>')
            
            elif line.startswith('### '):
                if in_list:
                    html_parts.append(f'</{list_type}>')
                    in_list = False
                subsection_text = line[4:].strip()
                # Extract emoji and text
                emoji_match = re.match(r'^([^\w\s]+)\s*\*?\*?(.+?)\*?\*?$', subsection_text)
                if emoji_match:
                    emoji, title = emoji_match.groups()
                    html_parts.append(f'<div class="subsection-header"><span class="subsection-emoji">{emoji}</span><h3 class="subsection-title">{title.strip()}</h3></div>')
                else:
                    html_parts.append(f'<h3 class="subsection-title">{subsection_text}</h3>')
            
            # Lists
            elif line.startswith('- ') or line.startswith('• '):
                if not in_list:
                    html_parts.append('<ul class="research-list">')
                    in_list = True
                    list_type = 'ul'
                list_content = line[2:].strip()
                # Handle bold/italic formatting
                list_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', list_content)
                list_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', list_content)
                html_parts.append(f'<li class="research-item">{list_content}</li>')
            
            elif re.match(r'^\d+\.\s+', line):
                if not in_list:
                    html_parts.append('<ol class="research-list numbered">')
                    in_list = True
                    list_type = 'ol'
                list_content = re.sub(r'^\d+\.\s+', '', line)
                # Handle bold/italic formatting
                list_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', list_content)
                list_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', list_content)
                html_parts.append(f'<li class="research-item">{list_content}</li>')
            
            # Horizontal rule
            elif line.startswith('---'):
                if in_list:
                    html_parts.append(f'</{list_type}>')
                    in_list = False
                html_parts.append('<hr class="section-divider"/>')
            
            # Next Steps highlight
            elif line.startswith('**Next Steps'):
                if in_list:
                    html_parts.append(f'</{list_type}>')
                    in_list = False
                next_steps_text = re.sub(r'\*\*(.+?)\*\*:', r'<strong>\1</strong>:', line)
                html_parts.append(f'<div class="next-steps-section">{next_steps_text}</div>')
            
            # Regular paragraphs
            else:
                if in_list:
                    html_parts.append(f'</{list_type}>')
                    in_list = False
                # Handle bold/italic formatting
                formatted_line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
                formatted_line = re.sub(r'\*(.+?)\*', r'<em>\1</em>', formatted_line)
                html_parts.append(f'<p class="research-text">{formatted_line}</p>')
        
        # Close any open lists
        if in_list:
            html_parts.append(f'</{list_type}>')
        
        # Add papers section if available
        if papers:
            html_parts.append('<div class="papers-showcase">')
            html_parts.append('<div class="section-header"><span class="section-emoji">📚</span><h2 class="section-title">Related Research Papers</h2></div>')
            
            for i, paper in enumerate(papers[:5]):  # Show top 5 papers
                html_parts.append('<div class="paper-card-enhanced">')
                html_parts.append(f'<div class="paper-rank">#{i+1}</div>')
                html_parts.append(f'<div class="paper-content">')
                html_parts.append(f'<h4 class="paper-title"><a href="https://arxiv.org/abs/{paper["arxiv_id"]}" target="_blank">{paper["title"]}</a></h4>')
                
                # Authors
                authors = paper.get("authors", [])
                authors_display = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_display += " et al."
                html_parts.append(f'<p class="paper-authors">👥 {authors_display}</p>')
                
                # Categories
                categories = paper.get("categories", [])[:3]
                html_parts.append('<div class="paper-tags">')
                for cat in categories:
                    html_parts.append(f'<span class="category-badge">{cat}</span>')
                html_parts.append('</div>')
                
                # Relevance score
                if paper.get("similarity", 0) > 0:
                    relevance = paper["similarity"] * 100
                    html_parts.append(f'<div class="relevance-indicator">🎯 {relevance:.0f}% relevant</div>')
                
                html_parts.append('</div>')  # paper-content
                html_parts.append('</div>')  # paper-card-enhanced
            
            html_parts.append('</div>')  # papers-showcase
        
        # Enhanced CSS styles
        html_parts.append('''
        <style>
            .research-assistant-response {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                line-height: 1.7;
                color: #2d3748;
                max-width: 100%;
                padding: 1rem 0;
            }
            
            .section-header, .subsection-header {
                display: flex;
                align-items: center;
                margin: 1.5rem 0 1rem 0;
                gap: 0.5rem;
            }
            
            .section-emoji, .subsection-emoji {
                font-size: 1.25em;
                filter: drop-shadow(0 1px 2px rgba(0,0,0,0.1));
            }
            
            .section-title {
                font-size: 1.25rem;
                font-weight: 700;
                color: #1a202c;
                margin: 0;
                border-bottom: 2px solid #e2e8f0;
                padding-bottom: 0.5rem;
                flex-grow: 1;
            }
            
            .subsection-title {
                font-size: 1.1rem;
                font-weight: 600;
                color: #2d3748;
                margin: 0;
            }
            
            .research-list {
                margin: 1rem 0;
                padding-left: 1.5rem;
            }
            
            .research-list.numbered {
                list-style-type: decimal;
            }
            
            .research-item {
                margin: 0.75rem 0;
                padding: 0.5rem 0;
                line-height: 1.6;
            }
            
            .research-item strong {
                color: #2b6cb0;
                font-weight: 600;
            }
            
            .research-text {
                margin: 1rem 0;
                line-height: 1.7;
            }
            
            .section-divider {
                border: none;
                height: 2px;
                background: linear-gradient(90deg, #4f46e5, #7c3aed, #4f46e5);
                margin: 2rem 0;
                border-radius: 1px;
            }
            
            .next-steps-section {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 12px;
                margin: 1.5rem 0;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
            
            .next-steps-section strong {
                color: #fef9e7;
            }
            
            .papers-showcase {
                margin-top: 2rem;
                padding: 1.5rem;
                background: linear-gradient(135deg, #f8faff 0%, #f1f5f9 100%);
                border-radius: 16px;
                border: 1px solid #e2e8f0;
            }
            
            .paper-card-enhanced {
                display: flex;
                align-items: flex-start;
                gap: 1rem;
                background: white;
                padding: 1.25rem;
                margin: 1rem 0;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                transition: all 0.2s ease;
            }
            
            .paper-card-enhanced:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.12);
                border-color: #3b82f6;
            }
            
            .paper-rank {
                background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                color: white;
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 0.9rem;
                flex-shrink: 0;
            }
            
            .paper-content {
                flex-grow: 1;
            }
            
            .paper-title {
                margin: 0 0 0.5rem 0;
                font-size: 1rem;
                line-height: 1.4;
            }
            
            .paper-title a {
                color: #1f2937;
                text-decoration: none;
                font-weight: 600;
            }
            
            .paper-title a:hover {
                color: #3b82f6;
                text-decoration: underline;
            }
            
            .paper-authors {
                color: #6b7280;
                font-size: 0.9rem;
                margin: 0.5rem 0;
            }
            
            .paper-tags {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin: 0.75rem 0;
            }
            
            .category-badge {
                background: #e0f2fe;
                color: #0277bd;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 500;
            }
            
            .relevance-indicator {
                background: #f0fdf4;
                color: #166534;
                padding: 0.25rem 0.75rem;
                border-radius: 6px;
                font-size: 0.85rem;
                font-weight: 500;
                margin-top: 0.5rem;
                display: inline-block;
            }
            
            /* Dark mode support */
            @media (prefers-color-scheme: dark) {
                .research-assistant-response {
                    color: #e2e8f0;
                }
                
                .section-title {
                    color: #f7fafc;
                    border-bottom-color: #4a5568;
                }
                
                .subsection-title {
                    color: #e2e8f0;
                }
                
                .papers-showcase {
                    background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
                    border-color: #4a5568;
                }
                
                .paper-card-enhanced {
                    background: #2d3748;
                    border-color: #4a5568;
                    color: #e2e8f0;
                }
                
                .paper-title a {
                    color: #e2e8f0;
                }
                
                .paper-title a:hover {
                    color: #63b3ed;
                }
            }
        </style>
        ''')
        
        html_parts.append('</div>')  # Close research-assistant-response
        
        return '\n'.join(html_parts)
    
    async def _generate_follow_up_questions(self, query: str, intent: Dict[str, Any]) -> List[str]:
        """
        Generate relevant follow-up questions based on the query and intent
        """
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": CHAT_AGENT_PROMPTS["follow_up"]},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        
        # Parse the response to get follow-up questions
        questions = completion.choices[0].message.content.split("\n")
        return [q.strip() for q in questions if q.strip()]
    
    def _extract_year(self, query: str) -> int:
        """
        Extract year from query if present
        """
        # Simple year extraction - can be made more sophisticated
        words = query.split()
        for word in words:
            if word.isdigit() and len(word) == 4:
                year = int(word)
                if 1900 <= year <= datetime.now().year:
                    return year
        return None
    
    def _extract_category(self, query: str) -> str:
        """
        Extract category from query if present
        """
        # Simple category extraction - can be made more sophisticated
        common_categories = ["cs", "math", "physics", "stat", "q-bio", "q-fin"]
        query_lower = query.lower()
        for category in common_categories:
            if category in query_lower:
                return category
        return None 