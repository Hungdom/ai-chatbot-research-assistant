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

    async def smart_search(self, query: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhanced search that uses category hierarchy, temporal patterns, and author analysis
        """
        try:
            self.logger.info(f"Starting smart search for: {query}")
            
            with get_db() as db:
                # Analyze query for smart parameters
                search_params = await self._analyze_query_smart(query, intent)
                self.logger.info(f"Smart search parameters: {json.dumps(search_params, indent=2)}")
                
                # Build enhanced query
                papers = await self._execute_smart_query(db, query, search_params)
                
                # Apply post-processing intelligence
                papers = await self._apply_smart_filtering(papers, search_params)
                
                self.logger.info(f"Smart search found {len(papers)} papers")
                return papers
                
        except Exception as e:
            self.logger.error(f"Error in smart search: {str(e)}", exc_info=True)
            return []

    async def _analyze_query_smart(self, query: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze query to extract smart search parameters
        """
        params = {
            "categories": [],
            "expanded_categories": [],
            "temporal_filters": {},
            "author_filters": [],
            "collaboration_patterns": [],
            "trend_analysis": False,
            "temporal_scope": "all",
            "interdisciplinary": False,
            "keywords": [],
            "methodologies": [],
            "research_level": "all"  # foundational, recent, emerging
        }
        
        query_lower = query.lower()
        
        # Category analysis
        params["categories"] = self._extract_categories_from_query(query)
        params["expanded_categories"] = self._expand_categories(params["categories"])
        
        # Temporal analysis
        params["temporal_filters"] = self._extract_temporal_info(query)
        
        # Detect if query is asking for trends or analysis
        if any(word in query_lower for word in ["trend", "evolution", "development", "progress", "growth"]):
            params["trend_analysis"] = True
            params["temporal_scope"] = "longitudinal"
        
        # Detect research level
        if any(word in query_lower for word in ["recent", "latest", "new", "current"]):
            params["research_level"] = "recent"
            params["temporal_filters"]["years"] = list(range(datetime.now().year - 2, datetime.now().year + 1))
        elif any(word in query_lower for word in ["foundational", "seminal", "classic", "fundamental"]):
            params["research_level"] = "foundational"
        
        # Detect interdisciplinary research
        if any(word in query_lower for word in ["interdisciplinary", "cross-field", "multi-field", "intersection"]):
            params["interdisciplinary"] = True
        
        # Extract methodologies
        params["methodologies"] = self._extract_methodologies(query)
        
        # Extract author patterns
        params["author_filters"] = self._extract_author_info(query)
        
        return params

    def _extract_categories_from_query(self, query: str) -> List[str]:
        """
        Extract categories from query using intelligent matching
        """
        categories = []
        query_lower = query.lower()
        
        # Direct category matching
        category_keywords = {
            "astro-ph": ["astrophysics", "astronomy", "cosmology", "galaxy", "star", "planet"],
            "cs.LG": ["machine learning", "deep learning", "neural network", "ai", "artificial intelligence"],
            "cs.CV": ["computer vision", "image recognition", "visual", "opencv"],
            "cs.CL": ["natural language", "nlp", "text processing", "linguistics"],
            "hep-ph": ["particle physics", "high energy", "standard model"],
            "gr-qc": ["general relativity", "gravity", "spacetime", "einstein"],
            "cond-mat": ["condensed matter", "materials", "solid state"],
            "quant-ph": ["quantum", "quantum mechanics", "entanglement"],
            "stat.ML": ["statistics", "statistical learning", "bayesian"],
            "physics.flu-dyn": ["fluid dynamics", "turbulence", "flow"],
            "math.OC": ["optimization", "control theory", "linear programming"],
            "q-bio": ["biology", "bioinformatics", "genetics", "protein"],
            "econ": ["economics", "finance", "market", "economic"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                categories.append(category)
        
        return categories

    def _expand_categories(self, categories: List[str]) -> List[str]:
        """
        Expand categories using hierarchy and relationships
        """
        expanded = set(categories)
        
        for category in categories:
            # Add subcategories
            if category in self.category_hierarchy:
                expanded.update(self.category_hierarchy[category])
            
            # Add related categories
            if category in self.category_relationships:
                expanded.update(self.category_relationships[category])
            
            # Handle parent categories
            for parent, children in self.category_hierarchy.items():
                if category in children:
                    expanded.add(parent)
        
        return list(expanded)

    def _extract_temporal_info(self, query: str) -> Dict[str, Any]:
        """
        Extract temporal information from query
        """
        temporal = {}
        
        # Extract years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = [int(year) for year in re.findall(year_pattern, query)]
        if years:
            temporal["years"] = years
        
        # Extract relative time periods
        query_lower = query.lower()
        current_year = datetime.now().year
        
        if any(phrase in query_lower for phrase in ["last year", "past year"]):
            temporal["years"] = [current_year - 1]
        elif any(phrase in query_lower for phrase in ["last 5 years", "past 5 years"]):
            temporal["years"] = list(range(current_year - 5, current_year + 1))
        elif any(phrase in query_lower for phrase in ["decade", "last decade"]):
            temporal["years"] = list(range(current_year - 10, current_year + 1))
        elif "this year" in query_lower:
            temporal["years"] = [current_year]
        
        return temporal

    def _extract_methodologies(self, query: str) -> List[str]:
        """
        Extract research methodologies from query
        """
        methodologies = []
        query_lower = query.lower()
        
        methodology_keywords = {
            "experimental": ["experiment", "empirical", "laboratory", "measurement"],
            "theoretical": ["theory", "theoretical", "model", "analytical"],
            "computational": ["simulation", "numerical", "computational", "algorithm"],
            "observational": ["observation", "survey", "data collection"],
            "statistical": ["statistical", "regression", "correlation", "analysis"],
            "review": ["review", "survey", "meta-analysis", "systematic"]
        }
        
        for methodology, keywords in methodology_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                methodologies.append(methodology)
        
        return methodologies

    def _extract_author_info(self, query: str) -> List[str]:
        """
        Extract author information from query
        """
        authors = []
        
        # Simple author name detection (can be enhanced with NER)
        author_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        potential_authors = re.findall(author_pattern, query)
        
        return potential_authors

    async def _execute_smart_query(self, db, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute the enhanced query with smart parameters
        """
        # Start with base query
        base_query = select(Arxiv)
        conditions = []
        
        # Category filtering with expansion
        if params["categories"] or params["expanded_categories"]:
            all_categories = set(params["categories"] + params["expanded_categories"])
            category_conditions = []
            
            for category in all_categories:
                # Handle exact matches and partial matches
                category_conditions.append(
                    or_(
                        Arxiv.categories.contains([category]),
                        Arxiv.primary_category == category,
                        text(f"EXISTS (SELECT 1 FROM unnest(categories) cat WHERE cat LIKE '%{category}%')")
                    )
                )
            
            if category_conditions:
                conditions.append(or_(*category_conditions))
        
        # Temporal filtering
        if params["temporal_filters"].get("years"):
            years = params["temporal_filters"]["years"]
            year_conditions = []
            for year in years:
                year_conditions.append(extract('year', Arxiv.published_date) == year)
            conditions.append(or_(*year_conditions))
        
        # Author filtering
        if params["author_filters"]:
            author_conditions = []
            for author in params["author_filters"]:
                author_conditions.append(
                    text(f"EXISTS (SELECT 1 FROM unnest(authors) auth WHERE auth ILIKE '%{author}%')")
                )
            conditions.append(or_(*author_conditions))
        
        # Keyword search in title and abstract
        keywords = query.split()
        if keywords:
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append(
                    or_(
                        Arxiv.title.ilike(f"%{keyword}%"),
                        Arxiv.abstract.ilike(f"%{keyword}%")
                    )
                )
            # For keyword search, we want papers that match most keywords
            if len(keyword_conditions) > 1:
                conditions.append(or_(*keyword_conditions))
            elif keyword_conditions:
                conditions.append(keyword_conditions[0])
        
        # Apply all conditions
        if conditions:
            base_query = base_query.where(and_(*conditions))
        
        # Handle embedding-based similarity if available
        query_embedding = await self._get_embedding(query)
        if query_embedding:
            embedding_check = db.execute(
                text("SELECT COUNT(*) FROM arxiv WHERE embedding IS NOT NULL")
            ).scalar()
            
            if embedding_check > 0:
                # Combine semantic and categorical search
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                similarity_expr = text(
                    f"cosine_similarity(arxiv.embedding, '{embedding_str}'::vector) as similarity"
                )
                
                # Create separate semantic query
                semantic_query = select(Arxiv, similarity_expr).select_from(Arxiv).where(
                    Arxiv.embedding.is_not(None)
                )
                
                # Apply same conditions to semantic query
                if conditions:
                    semantic_query = semantic_query.where(and_(*conditions))
                
                semantic_query = semantic_query.order_by(text("similarity DESC")).limit(20)
                semantic_results = db.execute(semantic_query).all()
                
                # Convert to papers with similarity scores
                papers = []
                for result in semantic_results:
                    if isinstance(result, tuple):
                        paper, similarity = result
                        if similarity and similarity > 0.1:  # Meaningful similarity threshold
                            papers.append(self._format_paper_with_similarity(paper, similarity))
                
                return papers
        
        # Fallback to categorical/keyword search
        base_query = base_query.order_by(Arxiv.published_date.desc()).limit(20)
        results = db.execute(base_query).scalars().all()
        
        papers = []
        for paper in results:
            papers.append(self._format_paper_dict(paper))
        
        return papers

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