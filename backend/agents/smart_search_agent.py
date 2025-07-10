#!/usr/bin/env python3
"""
Smart Search Agent - Enhanced search capabilities for ArXiv papers
Leverages hierarchical categories, temporal patterns, and author information
"""

import re
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from .base_agent import BaseAgent
from database import get_db, Arxiv
from sqlalchemy import text, func, desc, or_, and_, extract
from sqlalchemy.sql import select
import logging
import openai

class SmartSearchAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # ArXiv category hierarchy and relationships
        self.category_hierarchy = self._build_category_hierarchy()
        self.category_relationships = self._build_category_relationships()
        
    def _build_category_hierarchy(self) -> Dict[str, List[str]]:
        """
        Build category hierarchy from ArXiv categories
        """
        return {
            # Main categories and their subcategories
            "astro-ph": ["astro-ph.CO", "astro-ph.EP", "astro-ph.GA", "astro-ph.HE", 
                        "astro-ph.IM", "astro-ph.SR"],
            "cond-mat": ["cond-mat.dis-nn", "cond-mat.mes-hall", "cond-mat.mtrl-sci",
                        "cond-mat.other", "cond-mat.quant-gas", "cond-mat.soft",
                        "cond-mat.stat-mech", "cond-mat.str-el", "cond-mat.supr-con"],
            "cs": ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR",
                  "cs.CV", "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS",
                  "cs.ET", "cs.FL", "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR",
                  "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS", "cs.NA",
                  "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO",
                  "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"],
            "hep": ["hep-ex", "hep-lat", "hep-ph", "hep-th"],
            "gr-qc": [],
            "math": ["math.AC", "math.AG", "math.AP", "math.AT", "math.CA", "math.CO",
                    "math.CT", "math.CV", "math.DG", "math.DS", "math.FA", "math.GM",
                    "math.GN", "math.GR", "math.GT", "math.HO", "math.IT", "math.KT",
                    "math.LO", "math.MG", "math.MP", "math.NA", "math.NT", "math.OA",
                    "math.OC", "math.PR", "math.QA", "math.RA", "math.RT", "math.SG",
                    "math.SP", "math.ST"],
            "nucl": ["nucl-ex", "nucl-th"],
            "physics": ["physics.acc-ph", "physics.ao-ph", "physics.app-ph",
                       "physics.atm-clus", "physics.atom-ph", "physics.bio-ph",
                       "physics.chem-ph", "physics.class-ph", "physics.comp-ph",
                       "physics.data-an", "physics.ed-ph", "physics.flu-dyn",
                       "physics.gen-ph", "physics.geo-ph", "physics.hist-ph",
                       "physics.ins-det", "physics.med-ph", "physics.optics",
                       "physics.plasm-ph", "physics.pop-ph", "physics.soc-ph",
                       "physics.space-ph"],
            "quant-ph": [],
            "stat": ["stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.TH"],
            "q-bio": ["q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC",
                     "q-bio.OT", "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO"],
            "q-fin": ["q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM",
                     "q-fin.PR", "q-fin.RM", "q-fin.ST", "q-fin.TR"],
            "econ": ["econ.EM", "econ.GN", "econ.TH"],
            "eess": ["eess.AS", "eess.IV", "eess.SP", "eess.SY"]
        }
    
    def _build_category_relationships(self) -> Dict[str, Set[str]]:
        """
        Build relationships between categories based on common combinations
        """
        relationships = {
            # Common interdisciplinary combinations
            "astro-ph": {"gr-qc", "hep-ph", "hep-th", "cond-mat", "nucl-th"},
            "gr-qc": {"astro-ph", "hep-th", "hep-ph", "math-ph", "quant-ph"},
            "hep-ph": {"astro-ph", "gr-qc", "hep-th", "nucl-th", "physics.space-ph"},
            "hep-th": {"gr-qc", "hep-ph", "math-ph", "quant-ph", "astro-ph"},
            "cond-mat": {"astro-ph", "physics.atom-ph", "physics.flu-dyn", "physics.plasm-ph"},
            "cs.LG": {"stat.ML", "cs.AI", "cs.CV", "cs.CL"},
            "stat.ML": {"cs.LG", "cs.AI", "stat.TH", "stat.AP"},
            "physics.data-an": {"astro-ph", "stat.DA", "cs.LG"},
            "math.MP": {"hep-th", "gr-qc", "astro-ph", "cond-mat"},
            "quant-ph": {"gr-qc", "hep-th", "cond-mat", "physics.atom-ph"}
        }
        return relationships

    async def smart_search(self, query: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Main smart search method - TOKEN OPTIMIZED
        """
        try:
            self.logger.info(f"Smart search for: {query}")
            
            # 1. FAST PRELIMINARY ANALYSIS (avoid heavy AI calls)
            query_params = await self._quick_analyze_query(query, context)
            
            # 2. EXECUTE SMART SEARCH (efficient DB queries)
            papers = await self._execute_smart_query(query, query_params)
            
            # 3. CALCULATE SIMILARITY (optimized algorithm)
            if papers:
                papers_with_similarity = await self._calculate_smart_similarity(query, papers, query_params)
                
                # 4. SORT AND LIMIT RESULTS
                papers_with_similarity.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                
                # Keep only top results to avoid token overflow
                top_papers = papers_with_similarity[:15]  # Reduced from 20
                
                self.logger.info(f"Smart search completed: {len(top_papers)} papers with avg similarity: {sum(p.get('similarity', 0) for p in top_papers) / len(top_papers):.3f}")
                
                return top_papers
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in smart search: {str(e)}")
            return []

    async def _quick_analyze_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Quick query analysis without heavy AI calls
        """
        # Basic pattern matching instead of AI analysis
        query_lower = query.lower()
        
        # Detect categories from query text
        categories = []
        if any(term in query_lower for term in ['machine learning', 'ml', 'neural', 'deep learning', 'ai']):
            categories.extend(['cs.LG', 'cs.AI', 'stat.ML'])
        if any(term in query_lower for term in ['computer vision', 'cv', 'image', 'visual']):
            categories.extend(['cs.CV'])
        if any(term in query_lower for term in ['nlp', 'natural language', 'language model', 'text']):
            categories.extend(['cs.CL'])
        if any(term in query_lower for term in ['math', 'theorem', 'proof', 'algebra']):
            categories.extend(['math.AG', 'math.NT'])
        if any(term in query_lower for term in ['physics', 'quantum', 'particle']):
            categories.extend(['physics.gen-ph', 'quant-ph'])
        
        # Extract years from query
        years = []
        import re
        year_matches = re.findall(r'\b(19|20)\d{2}\b', query)
        years = [int(match[0] + match[1:]) for match in year_matches if match]
        
        # Extract potential author names (simple heuristic)
        authors = []
        author_patterns = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', query)
        authors = author_patterns[:3]  # Limit to 3 authors
        
        # Extract key terms (simple tokenization)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'about', 'what', 'how', 'when', 'where', 'why', 'which', 'who', 'find', 'search', 'look', 'paper', 'papers', 'research', 'study', 'studies'}
        
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        key_terms = [word for word in words if len(word) > 2 and word not in stop_words][:10]  # Top 10 terms
        
        return {
            "categories": categories,
            "years": years,
            "authors": authors,
            "key_terms": key_terms,
            "query_length": len(query),
            "complexity": "simple" if len(key_terms) <= 3 else "complex"
        }

    async def _execute_smart_query(self, query: str, params: Dict[str, Any]) -> List[Any]:
        """
        Execute optimized database query
        """
        try:
            with get_db() as db:
                # START WITH EFFICIENT BASE QUERY
                query_obj = select(Arxiv)
                
                # Add category filters if detected
                if params.get("categories"):
                    category_conditions = []
                    for category in params["categories"][:3]:  # Limit categories
                        category_conditions.append(Arxiv.categories.contains([category]))
                    if category_conditions:
                        query_obj = query_obj.where(or_(*category_conditions))
                
                # Add year filters if detected
                if params.get("years"):
                    year_conditions = []
                    for year in params["years"][:2]:  # Limit years
                        year_conditions.append(extract('year', Arxiv.published_date) == year)
                    if year_conditions:
                        query_obj = query_obj.where(or_(*year_conditions))
                
                # Add author filters if detected
                if params.get("authors"):
                    author_conditions = []
                    for author in params["authors"][:2]:  # Limit authors
                        author_conditions.append(
                            text(f"EXISTS (SELECT 1 FROM unnest(authors) auth WHERE auth ILIKE '%{author}%')")
                        )
                    if author_conditions:
                        query_obj = query_obj.where(or_(*author_conditions))
                
                # Add keyword search as fallback
                if params.get("key_terms") and not params.get("categories"):
                    # Use only top 3 keywords to avoid slow queries
                    top_terms = params["key_terms"][:3]
                    keyword_conditions = []
                    for term in top_terms:
                        keyword_conditions.append(
                            or_(
                                Arxiv.title.ilike(f'%{term}%'),
                                Arxiv.abstract.ilike(f'%{term}%')
                            )
                        )
                    if keyword_conditions:
                        query_obj = query_obj.where(or_(*keyword_conditions))
                
                # Order by publication date (recent first) and limit
                query_obj = query_obj.order_by(Arxiv.published_date.desc()).limit(25)  # Reduced from 50
                
                results = db.execute(query_obj).scalars().all()
                
                self.logger.info(f"Database query returned {len(results)} papers")
                return results
                
        except Exception as e:
            self.logger.error(f"Error executing smart query: {str(e)}")
            return []

    async def _calculate_smart_similarity(self, query: str, papers: List[Any], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate similarity using optimized algorithm
        """
        results = []
        query_lower = query.lower()
        query_terms = set(params.get("key_terms", []))
        
        for paper in papers:
            paper_dict = self._format_paper_dict(paper)
            
            # FAST SIMILARITY CALCULATION
            similarity = 0.0
            
            # 1. Title matching (40% weight)
            title_lower = paper_dict["title"].lower()
            title_words = set(re.findall(r'\b[a-zA-Z]+\b', title_lower))
            
            # Exact phrase matching
            if query_lower in title_lower:
                similarity += 0.25
            
            # Word overlap
            word_overlap = len(query_terms.intersection(title_words))
            if query_terms:
                similarity += (word_overlap / len(query_terms)) * 0.15
            
            # 2. Abstract matching (30% weight)
            abstract_lower = paper_dict["abstract"].lower()
            abstract_words = set(re.findall(r'\b[a-zA-Z]+\b', abstract_lower))
            
            if query_lower in abstract_lower:
                similarity += 0.15
            
            # Keyword density in abstract
            if query_terms:
                abstract_overlap = len(query_terms.intersection(abstract_words))
                similarity += (abstract_overlap / len(query_terms)) * 0.15
            
            # 3. Category matching (20% weight)
            if params.get("categories"):
                paper_categories = set(paper_dict.get("categories", []))
                param_categories = set(params["categories"])
                category_overlap = len(paper_categories.intersection(param_categories))
                if param_categories:
                    similarity += (category_overlap / len(param_categories)) * 0.20
            
            # 4. Author matching (10% weight)
            if params.get("authors"):
                paper_authors = " ".join(paper_dict.get("authors", [])).lower()
                author_matches = sum(1 for author in params["authors"] if author.lower() in paper_authors)
                if params["authors"]:
                    similarity += (author_matches / len(params["authors"])) * 0.10
            
            # 5. Recency bonus (small boost for recent papers)
            try:
                pub_year = int(paper_dict.get("published_date", "2000")[:4])
                current_year = datetime.now().year
                if pub_year >= current_year - 2:
                    similarity += 0.05
            except:
                pass
            
            # Only include papers with meaningful similarity
            if similarity > 0.1:  # Minimum threshold
                paper_dict["similarity"] = similarity
                results.append(paper_dict)
        
        return results

    async def _apply_smart_filtering(self, papers: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply intelligent post-processing filters
        """
        # Research level filtering
        if params["research_level"] == "foundational":
            # Prioritize highly cited or older influential papers
            papers = sorted(papers, key=lambda p: self._calculate_influence_score(p), reverse=True)
        elif params["research_level"] == "recent":
            # Prioritize recent papers
            papers = sorted(papers, key=lambda p: p.get("published_date", ""), reverse=True)
        
        # Interdisciplinary filtering
        if params["interdisciplinary"]:
            papers = [p for p in papers if len(p.get("categories", [])) > 2]
        
        # Trend analysis
        if params["trend_analysis"]:
            papers = await self._add_trend_analysis(papers)
        
        return papers

    def _calculate_influence_score(self, paper: Dict[str, Any]) -> float:
        """
        Calculate influence score based on various factors
        """
        score = 0.0
        
        # Age factor (older papers get slight boost for foundational search)
        try:
            pub_year = int(paper.get("published_date", "2000")[:4])
            age = 2024 - pub_year
            if age > 5:
                score += min(age * 0.1, 2.0)  # Cap at 2.0
        except:
            pass
        
        # Category diversity
        categories = paper.get("categories", [])
        score += len(categories) * 0.2
        
        # Abstract length (longer abstracts might indicate more comprehensive work)
        abstract_length = len(paper.get("abstract", ""))
        score += min(abstract_length / 1000, 1.0)
        
        return score

    async def _add_trend_analysis(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add trend analysis information to papers
        """
        # Group papers by year
        by_year = defaultdict(list)
        for paper in papers:
            try:
                year = int(paper.get("published_date", "2000")[:4])
                by_year[year].append(paper)
            except:
                continue
        
        # Add trend information
        for paper in papers:
            try:
                year = int(paper.get("published_date", "2000")[:4])
                paper["trend_info"] = {
                    "year": year,
                    "papers_in_year": len(by_year[year]),
                    "trend_direction": self._calculate_trend_direction(by_year, year)
                }
            except:
                paper["trend_info"] = {"year": None, "papers_in_year": 0, "trend_direction": "stable"}
        
        return papers

    def _calculate_trend_direction(self, by_year: Dict[int, List], current_year: int) -> str:
        """
        Calculate trend direction for a given year
        """
        if current_year - 1 in by_year and current_year + 1 in by_year:
            prev_count = len(by_year[current_year - 1])
            next_count = len(by_year[current_year + 1])
            current_count = len(by_year[current_year])
            
            if current_count > prev_count and current_count > next_count:
                return "peak"
            elif current_count < prev_count and current_count < next_count:
                return "valley"
            elif current_count > prev_count:
                return "increasing"
            elif current_count < prev_count:
                return "decreasing"
        
        return "stable"

    def _format_paper_with_similarity(self, paper: Arxiv, similarity: float) -> Dict[str, Any]:
        """
        Format paper with similarity score
        """
        paper_dict = self._format_paper_dict(paper)
        paper_dict["similarity"] = float(similarity)
        return paper_dict

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

    def _format_paper_dict(self, paper: Arxiv) -> Dict[str, Any]:
        """
        Format paper as dictionary
        """
        return {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "categories": paper.categories,
            "published_date": paper.published_date.isoformat() if paper.published_date else None,
            "doi": paper.doi,
            "primary_category": paper.primary_category,
            "similarity": 0.0
        }

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

    async def analyze_research_landscape(self, category: str, years: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyze research landscape for a given category
        """
        try:
            with get_db() as db:
                # Build query for landscape analysis
                query = select(Arxiv)
                
                # Category filter with expansion
                expanded_categories = self._expand_categories([category])
                category_conditions = []
                for cat in expanded_categories:
                    category_conditions.append(
                        or_(
                            Arxiv.categories.contains([cat]),
                            Arxiv.primary_category == cat
                        )
                    )
                
                if category_conditions:
                    query = query.where(or_(*category_conditions))
                
                # Year filter
                if years:
                    year_conditions = [extract('year', Arxiv.published_date) == year for year in years]
                    query = query.where(or_(*year_conditions))
                
                # Execute query
                papers = db.execute(query).scalars().all()
                
                # Analyze landscape
                analysis = {
                    "total_papers": len(papers),
                    "category_distribution": self._analyze_category_distribution(papers),
                    "temporal_trends": self._analyze_temporal_trends(papers),
                    "author_networks": self._analyze_author_networks(papers),
                    "collaboration_patterns": self._analyze_collaboration_patterns(papers),
                    "research_evolution": self._analyze_research_evolution(papers)
                }
                
                return analysis
                
        except Exception as e:
            self.logger.error(f"Error in landscape analysis: {str(e)}", exc_info=True)
            return {}

    def _analyze_category_distribution(self, papers: List[Arxiv]) -> Dict[str, int]:
        """
        Analyze distribution of categories
        """
        category_counts = Counter()
        for paper in papers:
            for category in paper.categories:
                category_counts[category] += 1
        
        return dict(category_counts.most_common(20))

    def _analyze_temporal_trends(self, papers: List[Arxiv]) -> Dict[str, Any]:
        """
        Analyze temporal trends in the papers
        """
        by_year = defaultdict(int)
        for paper in papers:
            if paper.published_date:
                year = paper.published_date.year
                by_year[year] += 1
        
        # Calculate growth rate
        years = sorted(by_year.keys())
        growth_rates = {}
        for i in range(1, len(years)):
            prev_year, curr_year = years[i-1], years[i]
            if by_year[prev_year] > 0:
                growth_rate = (by_year[curr_year] - by_year[prev_year]) / by_year[prev_year]
                growth_rates[curr_year] = growth_rate
        
        return {
            "papers_by_year": dict(by_year),
            "growth_rates": growth_rates,
            "peak_year": max(by_year.keys(), key=lambda y: by_year[y]) if by_year else None,
            "total_years": len(by_year)
        }

    def _analyze_author_networks(self, papers: List[Arxiv]) -> Dict[str, Any]:
        """
        Analyze author collaboration networks
        """
        author_counts = Counter()
        collaborations = defaultdict(set)
        
        for paper in papers:
            authors = paper.authors
            # Count individual authors
            for author in authors:
                author_counts[author] += 1
            
            # Track collaborations
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    collaborations[author1].add(author2)
                    collaborations[author2].add(author1)
        
        # Find most collaborative authors
        collaboration_scores = {
            author: len(collabs) for author, collabs in collaborations.items()
        }
        
        return {
            "top_authors": dict(author_counts.most_common(10)),
            "most_collaborative": dict(sorted(collaboration_scores.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]),
            "total_authors": len(author_counts),
            "average_collaborations": sum(collaboration_scores.values()) / len(collaboration_scores) if collaboration_scores else 0
        }

    def _analyze_collaboration_patterns(self, papers: List[Arxiv]) -> Dict[str, Any]:
        """
        Analyze collaboration patterns
        """
        team_sizes = []
        multi_category_papers = 0
        
        for paper in papers:
            team_sizes.append(len(paper.authors))
            if len(paper.categories) > 1:
                multi_category_papers += 1
        
        return {
            "average_team_size": sum(team_sizes) / len(team_sizes) if team_sizes else 0,
            "max_team_size": max(team_sizes) if team_sizes else 0,
            "min_team_size": min(team_sizes) if team_sizes else 0,
            "interdisciplinary_ratio": multi_category_papers / len(papers) if papers else 0,
            "team_size_distribution": dict(Counter(team_sizes))
        }

    def _analyze_research_evolution(self, papers: List[Arxiv]) -> Dict[str, Any]:
        """
        Analyze how research has evolved over time
        """
        # Group papers by time periods
        periods = defaultdict(list)
        for paper in papers:
            if paper.published_date:
                year = paper.published_date.year
                # Group into 5-year periods
                period = (year // 5) * 5
                periods[period].append(paper)
        
        # Analyze evolution of topics and categories
        evolution = {}
        for period, period_papers in periods.items():
            category_dist = Counter()
            for paper in period_papers:
                for category in paper.categories:
                    category_dist[category] += 1
            
            evolution[f"{period}s"] = {
                "paper_count": len(period_papers),
                "top_categories": dict(category_dist.most_common(5)),
                "avg_team_size": sum(len(p.authors) for p in period_papers) / len(period_papers)
            }
        
        return evolution 