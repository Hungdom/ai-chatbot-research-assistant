from typing import Dict, Any, List
from .base_agent import BaseAgent
import openai
from datetime import datetime
from config.prompts import CHAT_AGENT_PROMPTS
from database import get_db, Arxiv
from sqlalchemy import text, func, desc
from sqlalchemy.sql import select
import numpy as np
import json
import logging
from sqlalchemy.sql import or_
from sqlalchemy.sql import bindparam
import uuid
from database import ChatSession

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
        Search for relevant papers using both semantic and metadata search
        """
        try:
            self.logger.info("Starting paper search...")
            with get_db() as db:
                # Get query embedding
                query_embedding = await self._get_embedding(query)
                embedding_str = str(query_embedding).replace("'", "")

                # Build the query with similarity calculation
                similarity_expr = text(
                    "cosine_similarity(arxiv.embedding::vector, ARRAY" + embedding_str + "::vector) as similarity"
                )

                # Create the base query with filters
                query = select(
                    Arxiv,
                    similarity_expr
                ).select_from(Arxiv)

                # Add filters based on search parameters
                if intent.get("search_params", {}).get("year"):
                    year = intent["search_params"]["year"]
                    query = query.where(
                        extract('year', Arxiv.published_date) == year
                    )

                if intent.get("search_params", {}).get("category"):
                    category = intent["search_params"]["category"]
                    query = query.where(
                        Arxiv.categories.contains([category])
                    )

                # Add ordering and limit
                query = query.order_by(text("similarity DESC")).limit(10)

                # Execute query
                self.logger.info("Executing paper search query...")
                results = db.execute(query).all()
                
                # Convert results to dictionaries
                papers = []
                for paper, similarity in results:
                    paper_dict = {
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "abstract": paper.abstract,
                        "authors": paper.authors,
                        "categories": paper.categories,
                        "published_date": paper.published_date.isoformat() if paper.published_date else None,
                        "doi": paper.doi,
                        "primary_category": paper.primary_category,
                        "similarity": float(similarity) if similarity is not None else 0.0
                    }
                    papers.append(paper_dict)

                self.logger.info(f"Found {len(papers)} relevant papers")
                return papers

        except Exception as e:
            self.logger.error(f"Error searching papers: {str(e)}", exc_info=True)
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
    
    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using OpenAI API
        """
        try:
            self.logger.info(f"Generating embedding for text: {text[:100]}...")
            response = await openai.Embedding.acreate(
                input=text,
                model="text-embedding-ada-002"
            )
            self.logger.info("Embedding generated successfully")
            return response.data[0].embedding
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
    
    async def _generate_response(self, query: str, intent: Dict[str, Any], arxiv_results: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a response based on the query, intent, and search results
        """
        try:
            # Prepare context for the AI
            response_context = {
                "query": query,
                "intent_type": intent.get("type", "general_query"),
                "has_papers": len(arxiv_results) > 0,
                "papers": arxiv_results,
                "context": context,
                "search_params": intent.get("search_params", {}),
                "follow_up_questions": intent.get("follow_up_questions", [])
            }
            
            # Select appropriate prompt template
            if intent.get("type") == "paper_search":
                if arxiv_results:
                    prompt = CHAT_AGENT_PROMPTS["paper_search_with_results"]
                else:
                    prompt = CHAT_AGENT_PROMPTS["paper_search_no_results"]
            else:
                prompt = CHAT_AGENT_PROMPTS["general_query"]
            
            # Format the prompt with safe string formatting
            try:
                formatted_prompt = prompt.format(
                    query=response_context["query"],
                    intent_type=response_context["intent_type"],
                    has_papers=response_context["has_papers"],
                    papers=json.dumps(response_context["papers"], indent=2),
                    context=json.dumps(response_context["context"], indent=2),
                    search_params=json.dumps(response_context["search_params"], indent=2),
                    follow_up_questions=json.dumps(response_context["follow_up_questions"], indent=2)
                )
            except KeyError as e:
                self.logger.error(f"Error formatting prompt: {str(e)}")
                formatted_prompt = f"Query: {query}\nIntent: {intent.get('type', 'general_query')}\nResults: {len(arxiv_results)} papers found"
            
            # Generate completion
            completion = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": CHAT_AGENT_PROMPTS["system"]},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.7
            )
            
            response_text = completion.choices[0].message.content
            
            # Generate HTML version of the response
            html_response = self._format_response_as_html(response_text, arxiv_results, intent)
            
            return {
                "response": response_text,
                "html_response": html_response,
                "arxiv": arxiv_results,
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
        Format the response as HTML with proper styling and structure
        """
        html_parts = ['<div class="chat-response">']
        
        # Add main response text with proper formatting
        html_parts.append('<div class="response-text">')
        # Convert markdown-style formatting to HTML
        formatted_text = response_text.replace('**', '<strong>').replace('*', '<em>')
        html_parts.append(f'<p>{formatted_text}</p>')
        html_parts.append('</div>')
        
        # Add papers if available
        if papers:
            html_parts.append('<div class="papers-section">')
            html_parts.append('<h3 class="section-title">📚 Relevant Papers</h3>')
            html_parts.append('<div class="papers-list">')
            
            for paper in papers:
                html_parts.append('<div class="paper-card">')
                # Paper title with link to arXiv
                html_parts.append(f'<h4 class="paper-title"><a href="https://arxiv.org/abs/{paper["arxiv_id"]}" target="_blank">{paper["title"]}</a></h4>')
                
                # Authors with icons
                html_parts.append('<div class="paper-authors">')
                html_parts.append('<i class="fas fa-users"></i> ')
                html_parts.append(f'<span>{", ".join(paper["authors"])}</span>')
                html_parts.append('</div>')
                
                # Categories with tags
                html_parts.append('<div class="paper-categories">')
                for category in paper["categories"]:
                    html_parts.append(f'<span class="category-tag">{category}</span>')
                html_parts.append('</div>')
                
                # Abstract with expandable section
                html_parts.append('<div class="paper-abstract">')
                html_parts.append('<details>')
                html_parts.append('<summary>Abstract</summary>')
                html_parts.append(f'<p>{paper["abstract"]}</p>')
                html_parts.append('</details>')
                html_parts.append('</div>')
                
                # Metadata section
                html_parts.append('<div class="paper-metadata">')
                if paper.get("published_date"):
                    html_parts.append(f'<span class="date"><i class="far fa-calendar"></i> {paper["published_date"]}</span>')
                if paper.get("doi"):
                    html_parts.append(f'<span class="doi"><i class="fas fa-link"></i> <a href="https://doi.org/{paper["doi"]}" target="_blank">DOI</a></span>')
                html_parts.append('</div>')
                
                # Similarity score if available
                if "similarity" in paper:
                    similarity = float(paper["similarity"])
                    html_parts.append(f'<div class="similarity-score">Relevance: {similarity:.2%}</div>')
                
                html_parts.append('</div>')  # Close paper-card
            html_parts.append('</div>')  # Close papers-list
            html_parts.append('</div>')  # Close papers-section
        
        # Add insights if available
        if intent.get("insights"):
            html_parts.append('<div class="insights-section">')
            html_parts.append('<h3 class="section-title">📊 Research Insights</h3>')
            
            if intent["insights"].get("trends"):
                html_parts.append('<div class="trends">')
                html_parts.append('<h4><i class="fas fa-chart-line"></i> Research Trends</h4>')
                for trend_type, trend_data in intent["insights"]["trends"].items():
                    html_parts.append(f'<div class="trend-category"><h5>{trend_type.title()}</h5>')
                    html_parts.append('<ul class="trend-list">')
                    for item, count in trend_data.items():
                        html_parts.append(f'<li><span class="trend-item">{item}</span> <span class="trend-count">({count})</span></li>')
                    html_parts.append('</ul></div>')
                html_parts.append('</div>')
            
            if intent["insights"].get("gaps"):
                html_parts.append('<div class="research-gaps">')
                html_parts.append('<h4><i class="fas fa-lightbulb"></i> Research Gaps</h4>')
                html_parts.append('<ul class="gaps-list">')
                for gap in intent["insights"]["gaps"]:
                    html_parts.append(f'<li>{gap}</li>')
                html_parts.append('</ul>')
                html_parts.append('</div>')
            
            html_parts.append('</div>')  # Close insights-section
        
        # Add follow-up questions if available
        if intent.get("follow_up_questions"):
            html_parts.append('<div class="follow-up-section">')
            html_parts.append('<h3 class="section-title">💡 Suggested Follow-up Questions</h3>')
            html_parts.append('<ul class="follow-up-list">')
            for question in intent["follow_up_questions"]:
                html_parts.append(f'<li><i class="fas fa-arrow-right"></i> {question}</li>')
            html_parts.append('</ul>')
            html_parts.append('</div>')
        
        # Add CSS styles
        html_parts.append('''
        <style>
            .chat-response {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                line-height: 1.6;
                color: #333;
            }
            .section-title {
                color: #2c3e50;
                border-bottom: 2px solid #eee;
                padding-bottom: 0.5em;
                margin-top: 1.5em;
            }
            .paper-card {
                background: #fff;
                border: 1px solid #e1e4e8;
                border-radius: 6px;
                padding: 1.5em;
                margin: 1em 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .paper-title {
                margin: 0 0 0.5em;
                font-size: 1.2em;
            }
            .paper-title a {
                color: #0366d6;
                text-decoration: none;
            }
            .paper-title a:hover {
                text-decoration: underline;
            }
            .paper-authors {
                color: #586069;
                margin-bottom: 0.5em;
            }
            .category-tag {
                display: inline-block;
                background: #e1ecf4;
                color: #39739d;
                padding: 0.2em 0.6em;
                border-radius: 3px;
                margin: 0.2em;
                font-size: 0.9em;
            }
            .paper-abstract {
                margin: 1em 0;
            }
            .paper-metadata {
                color: #586069;
                font-size: 0.9em;
                margin-top: 1em;
            }
            .paper-metadata span {
                margin-right: 1em;
            }
            .similarity-score {
                background: #f1f8ff;
                color: #0366d6;
                padding: 0.3em 0.6em;
                border-radius: 3px;
                display: inline-block;
                margin-top: 0.5em;
            }
            .trend-list, .gaps-list, .follow-up-list {
                list-style: none;
                padding-left: 0;
            }
            .trend-item {
                color: #0366d6;
            }
            .trend-count {
                color: #586069;
                font-size: 0.9em;
            }
            .follow-up-list li {
                margin: 0.5em 0;
                padding-left: 1.5em;
                position: relative;
            }
            .follow-up-list li i {
                position: absolute;
                left: 0;
                color: #0366d6;
            }
        </style>
        ''')
        
        html_parts.append('</div>')  # Close chat-response
        
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