#!/usr/bin/env python3
"""
Enhanced Chat Agent - Integrates smart search capabilities
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from .smart_search_agent import SmartSearchAgent
from database import get_db, Arxiv, ChatSession
from sqlalchemy import text, func, desc, or_, extract
import openai
import json
import uuid
from datetime import datetime
import logging

class EnhancedChatAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.smart_search = SmartSearchAgent()
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user input with enhanced intelligence
        """
        try:
            self.logger.info("=== Enhanced Chat Processing ===")
            
            query = input_data.get("query", "")
            context = input_data.get("context", {})
            session_id = input_data.get("session_id")
            
            # Enhanced intent analysis
            intent = await self._analyze_intent_enhanced(query)
            self.logger.info(f"Enhanced intent: {json.dumps(intent, indent=2)}")
            
            # Smart search if needed
            arxiv_results = []
            if intent.get("needs_search", False):
                arxiv_results = await self._execute_enhanced_search(query, intent)
                
                # Add research insights
                if arxiv_results:
                    insights = await self._generate_enhanced_insights(arxiv_results, intent)
                    intent["insights"] = insights
            
            # Generate enhanced response
            response_data = await self._generate_enhanced_response(query, intent, arxiv_results, context)
            
            # Store session if needed
            if session_id:
                await self._update_session(session_id, query, response_data)
            
            return {
                "response": response_data.get("response", ""),
                "html_response": response_data.get("html_response", ""),
                "arxiv": arxiv_results,
                "intent": intent,
                "session_id": session_id,
                "metadata": {
                    "search_strategy": intent.get("search_strategy", "basic"),
                    "papers_found": len(arxiv_results),
                    "research_areas": intent.get("research_areas", []),
                    "temporal_scope": intent.get("temporal_scope", "all")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in enhanced chat processing: {str(e)}", exc_info=True)
            return self._generate_error_response(query, session_id)

    async def _analyze_intent_enhanced(self, query: str) -> Dict[str, Any]:
        """
        Enhanced intent analysis using multiple strategies
        """
        # Basic intent analysis
        completion = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an advanced research assistant. Analyze the user's query and return a JSON response with:
                    1. query_type: paper_search, trend_analysis, author_analysis, category_exploration, comparison, general_question
                    2. research_areas: list of identified research areas/categories
                    3. temporal_scope: recent, historical, longitudinal, specific_year, all
                    4. specificity_level: broad, focused, very_specific
                    5. needs_search: boolean
                    6. search_strategy: semantic, categorical, temporal, author_based, hybrid
                    7. complexity_level: simple, moderate, complex
                    8. expected_answer_type: papers, trends, analysis, explanation, comparison
                    """
                },
                {"role": "user", "content": f"Analyze this research query: {query}"}
            ],
            temperature=0.3
        )
        
        try:
            intent = json.loads(completion.choices[0].message.content)
        except:
            # Fallback intent
            intent = {
                "query_type": "general_question",
                "research_areas": [],
                "temporal_scope": "all",
                "specificity_level": "broad",
                "needs_search": True,
                "search_strategy": "hybrid",
                "complexity_level": "moderate",
                "expected_answer_type": "papers"
            }
        
        # Add smart parameter analysis
        smart_params = await self.smart_search._analyze_query_smart(query, intent)
        intent.update(smart_params)
        
        return intent

    async def _execute_enhanced_search(self, query: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute search using enhanced strategies
        """
        search_strategy = intent.get("search_strategy", "hybrid")
        
        if search_strategy == "hybrid":
            # Use smart search for most queries
            return await self.smart_search.smart_search(query, intent)
        elif search_strategy == "categorical":
            return await self._categorical_search(query, intent)
        elif search_strategy == "temporal":
            return await self._temporal_search(query, intent)
        elif search_strategy == "author_based":
            return await self._author_search(query, intent)
        else:
            # Default to smart search
            return await self.smart_search.smart_search(query, intent)

    async def _categorical_search(self, query: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Category-focused search
        """
        try:
            with get_db() as db:
                categories = intent.get("categories", []) + intent.get("expanded_categories", [])
                
                if not categories:
                    return []
                
                # Build category-focused query
                category_conditions = []
                for category in categories:
                    category_conditions.append(
                        or_(
                            Arxiv.categories.contains([category]),
                            Arxiv.primary_category == category
                        )
                    )
                
                query_obj = select(Arxiv).where(or_(*category_conditions))
                
                # Add temporal filters if specified
                if intent.get("temporal_filters", {}).get("years"):
                    years = intent["temporal_filters"]["years"]
                    year_conditions = [extract('year', Arxiv.published_date) == year for year in years]
                    query_obj = query_obj.where(or_(*year_conditions))
                
                query_obj = query_obj.order_by(Arxiv.published_date.desc()).limit(15)
                results = db.execute(query_obj).scalars().all()
                
                return [self.smart_search._format_paper_dict(paper) for paper in results]
                
        except Exception as e:
            self.logger.error(f"Error in categorical search: {str(e)}")
            return []

    async def _temporal_search(self, query: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Time-focused search for trend analysis
        """
        try:
            with get_db() as db:
                temporal_filters = intent.get("temporal_filters", {})
                
                # Build temporal query
                query_obj = select(Arxiv)
                
                if temporal_filters.get("years"):
                    years = temporal_filters["years"]
                    year_conditions = [extract('year', Arxiv.published_date) == year for year in years]
                    query_obj = query_obj.where(or_(*year_conditions))
                
                # Add category filters if available
                categories = intent.get("categories", [])
                if categories:
                    category_conditions = []
                    for category in categories:
                        category_conditions.append(Arxiv.categories.contains([category]))
                    query_obj = query_obj.where(or_(*category_conditions))
                
                # Order by date for temporal analysis
                query_obj = query_obj.order_by(Arxiv.published_date.asc()).limit(20)
                results = db.execute(query_obj).scalars().all()
                
                papers = [self.smart_search._format_paper_dict(paper) for paper in results]
                
                # Add temporal analysis
                return await self.smart_search._add_trend_analysis(papers)
                
        except Exception as e:
            self.logger.error(f"Error in temporal search: {str(e)}")
            return []

    async def _author_search(self, query: str, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Author-focused search
        """
        try:
            with get_db() as db:
                author_filters = intent.get("author_filters", [])
                
                if not author_filters:
                    return []
                
                # Build author-focused query
                author_conditions = []
                for author in author_filters:
                    author_conditions.append(
                        text(f"EXISTS (SELECT 1 FROM unnest(authors) auth WHERE auth ILIKE '%{author}%')")
                    )
                
                query_obj = select(Arxiv).where(or_(*author_conditions))
                query_obj = query_obj.order_by(Arxiv.published_date.desc()).limit(15)
                results = db.execute(query_obj).scalars().all()
                
                return [self.smart_search._format_paper_dict(paper) for paper in results]
                
        except Exception as e:
            self.logger.error(f"Error in author search: {str(e)}")
            return []

    async def _generate_enhanced_insights(self, papers: List[Dict[str, Any]], intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate enhanced insights based on intent and papers
        """
        insights = {}
        
        # Basic insights
        insights.update(await self._generate_basic_insights(papers))
        
        # Query-specific insights
        query_type = intent.get("query_type", "paper_search")
        
        if query_type == "trend_analysis":
            insights["trend_analysis"] = await self._analyze_trends_detailed(papers)
        elif query_type == "author_analysis":
            insights["author_analysis"] = await self._analyze_authors_detailed(papers)
        elif query_type == "category_exploration":
            insights["category_analysis"] = await self._analyze_categories_detailed(papers)
        elif query_type == "comparison":
            insights["comparison_analysis"] = await self._generate_comparison_insights(papers)
        
        # Research landscape if enough papers
        if len(papers) > 5:
            insights["research_landscape"] = await self._analyze_research_landscape_summary(papers)
        
        return insights

    async def _analyze_trends_detailed(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detailed trend analysis
        """
        from collections import defaultdict
        
        # Group by year
        by_year = defaultdict(list)
        for paper in papers:
            try:
                year = int(paper.get("published_date", "2000")[:4])
                by_year[year].append(paper)
            except:
                continue
        
        # Calculate trends
        years = sorted(by_year.keys())
        paper_counts = [len(by_year[year]) for year in years]
        
        # Growth analysis
        growth_periods = []
        for i in range(1, len(paper_counts)):
            if paper_counts[i] > paper_counts[i-1]:
                growth_periods.append(f"{years[i-1]}-{years[i]}")
        
        # Category evolution
        category_evolution = {}
        for year in years[-3:]:  # Last 3 years
            year_papers = by_year[year]
            categories = []
            for paper in year_papers:
                categories.extend(paper.get("categories", []))
            
            from collections import Counter
            category_evolution[year] = dict(Counter(categories).most_common(5))
        
        return {
            "paper_count_by_year": dict(zip(years, paper_counts)),
            "growth_periods": growth_periods,
            "category_evolution": category_evolution,
            "peak_year": years[paper_counts.index(max(paper_counts))] if paper_counts else None,
            "trend_direction": "increasing" if len(growth_periods) > len(years)//2 else "stable"
        }

    async def _analyze_authors_detailed(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detailed author analysis
        """
        from collections import Counter, defaultdict
        
        author_papers = defaultdict(list)
        collaborations = defaultdict(set)
        
        for paper in papers:
            authors = paper.get("authors", [])
            for author in authors:
                author_papers[author].append(paper)
            
            # Track collaborations
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    collaborations[author1].add(author2)
                    collaborations[author2].add(author1)
        
        # Top authors by paper count
        top_authors = dict(Counter({author: len(papers) for author, papers in author_papers.items()}).most_common(10))
        
        # Most collaborative authors
        collaboration_scores = {author: len(collabs) for author, collabs in collaborations.items()}
        top_collaborators = dict(sorted(collaboration_scores.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Research evolution by author
        author_evolution = {}
        for author, author_paper_list in list(author_papers.items())[:5]:  # Top 5 authors
            years = []
            for paper in author_paper_list:
                try:
                    year = int(paper.get("published_date", "2000")[:4])
                    years.append(year)
                except:
                    continue
            
            if years:
                author_evolution[author] = {
                    "first_paper": min(years),
                    "latest_paper": max(years),
                    "active_years": max(years) - min(years) + 1,
                    "paper_count": len(author_paper_list)
                }
        
        return {
            "top_authors": top_authors,
            "top_collaborators": top_collaborators,
            "author_evolution": author_evolution,
            "total_unique_authors": len(author_papers),
            "average_collaboration_network": sum(collaboration_scores.values()) / len(collaboration_scores) if collaboration_scores else 0
        }

    async def _analyze_categories_detailed(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detailed category analysis
        """
        from collections import Counter, defaultdict
        
        # Category frequency
        all_categories = []
        category_combinations = []
        interdisciplinary_papers = 0
        
        for paper in papers:
            categories = paper.get("categories", [])
            all_categories.extend(categories)
            
            if len(categories) > 1:
                interdisciplinary_papers += 1
                category_combinations.append(tuple(sorted(categories)))
        
        category_freq = dict(Counter(all_categories).most_common(15))
        combination_freq = dict(Counter(category_combinations).most_common(10))
        
        # Category hierarchy analysis
        main_categories = defaultdict(int)
        for category in all_categories:
            if '.' in category:
                main_cat = category.split('.')[0]
            elif '-' in category:
                main_cat = category.split('-')[0]
            else:
                main_cat = category
            main_categories[main_cat] += 1
        
        return {
            "category_frequency": category_freq,
            "main_category_distribution": dict(main_categories),
            "common_combinations": combination_freq,
            "interdisciplinary_ratio": interdisciplinary_papers / len(papers) if papers else 0,
            "category_diversity": len(set(all_categories)),
            "average_categories_per_paper": len(all_categories) / len(papers) if papers else 0
        }

    async def _generate_comparison_insights(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comparison insights between different aspects
        """
        # Compare by time periods
        current_year = datetime.now().year
        recent_papers = [p for p in papers if int(p.get("published_date", "2000")[:4]) >= current_year - 3]
        older_papers = [p for p in papers if int(p.get("published_date", "2000")[:4]) < current_year - 3]
        
        # Category comparison
        recent_categories = []
        older_categories = []
        
        for paper in recent_papers:
            recent_categories.extend(paper.get("categories", []))
        
        for paper in older_papers:
            older_categories.extend(paper.get("categories", []))
        
        from collections import Counter
        recent_cat_freq = dict(Counter(recent_categories).most_common(10))
        older_cat_freq = dict(Counter(older_categories).most_common(10))
        
        return {
            "temporal_comparison": {
                "recent_papers": len(recent_papers),
                "older_papers": len(older_papers),
                "recent_top_categories": recent_cat_freq,
                "older_top_categories": older_cat_freq
            },
            "research_shift": {
                "emerging_categories": [cat for cat in recent_cat_freq if cat not in older_cat_freq],
                "declining_categories": [cat for cat in older_cat_freq if cat not in recent_cat_freq]
            }
        }

    async def _analyze_research_landscape_summary(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of the research landscape
        """
        # Extract key patterns
        total_papers = len(papers)
        
        # Time span
        years = []
        for paper in papers:
            try:
                year = int(paper.get("published_date", "2000")[:4])
                years.append(year)
            except:
                continue
        
        time_span = max(years) - min(years) + 1 if years else 0
        
        # Research intensity (papers per year)
        research_intensity = total_papers / time_span if time_span > 0 else 0
        
        # Interdisciplinary index
        multi_category_papers = sum(1 for p in papers if len(p.get("categories", [])) > 1)
        interdisciplinary_index = multi_category_papers / total_papers if total_papers > 0 else 0
        
        # Collaboration index
        total_authors = sum(len(p.get("authors", [])) for p in papers)
        avg_team_size = total_authors / total_papers if total_papers > 0 else 0
        
        return {
            "research_maturity": "emerging" if time_span < 5 else "established" if time_span < 15 else "mature",
            "research_intensity": research_intensity,
            "interdisciplinary_index": interdisciplinary_index,
            "collaboration_level": "low" if avg_team_size < 3 else "high" if avg_team_size > 6 else "moderate",
            "time_span_years": time_span,
            "research_activity": "active" if research_intensity > 10 else "moderate" if research_intensity > 5 else "low"
        }

    async def _generate_basic_insights(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate basic insights about the papers
        """
        if not papers:
            return {}
        
        # Basic statistics
        total_papers = len(papers)
        
        # Year distribution
        years = []
        for paper in papers:
            try:
                year = int(paper.get("published_date", "2000")[:4])
                years.append(year)
            except:
                continue
        
        # Category distribution
        all_categories = []
        for paper in papers:
            all_categories.extend(paper.get("categories", []))
        
        from collections import Counter
        top_categories = dict(Counter(all_categories).most_common(10))
        
        # Author statistics
        all_authors = []
        for paper in papers:
            all_authors.extend(paper.get("authors", []))
        
        unique_authors = len(set(all_authors))
        top_authors = dict(Counter(all_authors).most_common(5))
        
        return {
            "total_papers": total_papers,
            "year_range": {"min": min(years), "max": max(years)} if years else {},
            "top_categories": top_categories,
            "top_authors": top_authors,
            "unique_authors": unique_authors,
            "average_authors_per_paper": len(all_authors) / total_papers if total_papers > 0 else 0
        }

    async def _generate_enhanced_response(self, query: str, intent: Dict[str, Any], papers: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate enhanced response with rich insights
        """
        # Prepare enhanced context
        response_context = {
            "query": query,
            "intent": intent,
            "papers": papers,
            "context": context,
            "insights": intent.get("insights", {}),
            "metadata": {
                "papers_found": len(papers),
                "search_strategy": intent.get("search_strategy", "basic"),
                "query_complexity": intent.get("complexity_level", "moderate")
            }
        }
        
        # Generate response using GPT-4 for better quality
        system_prompt = """You are an advanced research assistant with deep knowledge of academic research. 
        Generate comprehensive, insightful responses that:
        1. Address the user's specific query
        2. Provide relevant paper recommendations with context
        3. Include insights about research trends and patterns
        4. Suggest follow-up research directions
        5. Explain the significance of findings
        
        Format your response to be informative, well-structured, and engaging."""
        
        user_prompt = f"""
        Query: {query}
        
        Search Results: {len(papers)} papers found
        
        Research Insights: {json.dumps(intent.get('insights', {}), indent=2)}
        
        Query Analysis: {json.dumps(intent, indent=2)}
        
        Please provide a comprehensive response addressing the user's research query.
        """
        
        try:
            completion = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            response_text = completion.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            response_text = self._generate_fallback_response(query, papers, intent)
        
        # Generate HTML response
        html_response = await self._format_enhanced_html_response(response_text, papers, intent)
        
        return {
            "response": response_text,
            "html_response": html_response
        }

    def _generate_fallback_response(self, query: str, papers: List[Dict[str, Any]], intent: Dict[str, Any]) -> str:
        """
        Generate a fallback response when GPT fails
        """
        if not papers:
            return f"I couldn't find specific papers for '{query}', but I can help you explore related research areas or refine your search."
        
        response_parts = [
            f"I found {len(papers)} relevant papers for your query about '{query}'.",
            "",
            "Key findings from the research:"
        ]
        
        # Add insights if available
        insights = intent.get("insights", {})
        if insights.get("top_categories"):
            categories = list(insights["top_categories"].keys())[:3]
            response_parts.append(f"• Main research areas: {', '.join(categories)}")
        
        if insights.get("year_range"):
            year_range = insights["year_range"]
            response_parts.append(f"• Research spans from {year_range.get('min', 'N/A')} to {year_range.get('max', 'N/A')}")
        
        response_parts.extend([
            "",
            "I can help you explore these papers further or analyze specific aspects of this research area."
        ])
        
        return "\n".join(response_parts)

    async def _format_enhanced_html_response(self, response_text: str, papers: List[Dict[str, Any]], intent: Dict[str, Any]) -> str:
        """
        Format enhanced HTML response with rich visualizations
        """
        html_parts = ['<div class="enhanced-chat-response">']
        
        # Response text
        html_parts.append('<div class="response-text">')
        html_parts.append(f'<p>{response_text.replace(chr(10), "</p><p>")}</p>')
        html_parts.append('</div>')
        
        # Papers section with enhanced metadata
        if papers:
            html_parts.append('<div class="papers-section">')
            html_parts.append(f'<h3 class="section-title">📚 Found {len(papers)} Relevant Papers</h3>')
            
            for i, paper in enumerate(papers[:10]):  # Limit to top 10
                html_parts.append('<div class="enhanced-paper-card">')
                
                # Paper header with ranking
                html_parts.append(f'<div class="paper-header">')
                html_parts.append(f'<span class="paper-rank">#{i+1}</span>')
                html_parts.append(f'<h4 class="paper-title"><a href="https://arxiv.org/abs/{paper["arxiv_id"]}" target="_blank">{paper["title"]}</a></h4>')
                html_parts.append('</div>')
                
                # Enhanced metadata
                html_parts.append('<div class="paper-metadata-enhanced">')
                html_parts.append(f'<div class="metadata-row">')
                html_parts.append(f'<span class="metadata-label">Authors:</span>')
                html_parts.append(f'<span class="metadata-value">{", ".join(paper["authors"][:3])}{" et al." if len(paper["authors"]) > 3 else ""}</span>')
                html_parts.append('</div>')
                
                html_parts.append(f'<div class="metadata-row">')
                html_parts.append(f'<span class="metadata-label">Published:</span>')
                html_parts.append(f'<span class="metadata-value">{paper.get("published_date", "N/A")[:10]}</span>')
                html_parts.append('</div>')
                
                if paper.get("similarity", 0) > 0:
                    similarity = paper["similarity"]
                    html_parts.append(f'<div class="metadata-row">')
                    html_parts.append(f'<span class="metadata-label">Relevance:</span>')
                    html_parts.append(f'<span class="relevance-score">{similarity:.1%}</span>')
                    html_parts.append('</div>')
                
                html_parts.append('</div>')
                
                # Categories with enhanced styling
                html_parts.append('<div class="enhanced-categories">')
                for category in paper.get("categories", [])[:5]:
                    category_class = self._get_category_class(category)
                    html_parts.append(f'<span class="category-tag {category_class}">{category}</span>')
                html_parts.append('</div>')
                
                # Abstract preview
                abstract = paper.get("abstract", "")
                if abstract:
                    preview = abstract[:200] + "..." if len(abstract) > 200 else abstract
                    html_parts.append('<div class="abstract-preview">')
                    html_parts.append(f'<p>{preview}</p>')
                    html_parts.append('</div>')
                
                html_parts.append('</div>')  # Close enhanced-paper-card
            
            html_parts.append('</div>')  # Close papers-section
        
        # Enhanced insights section
        insights = intent.get("insights", {})
        if insights:
            html_parts.append('<div class="enhanced-insights-section">')
            html_parts.append('<h3 class="section-title">📊 Research Insights</h3>')
            
            # Research landscape
            if insights.get("research_landscape"):
                landscape = insights["research_landscape"]
                html_parts.append('<div class="landscape-summary">')
                html_parts.append('<h4>🌍 Research Landscape</h4>')
                html_parts.append(f'<div class="landscape-metrics">')
                html_parts.append(f'<span class="metric">Maturity: <strong>{landscape.get("research_maturity", "N/A")}</strong></span>')
                html_parts.append(f'<span class="metric">Activity: <strong>{landscape.get("research_activity", "N/A")}</strong></span>')
                html_parts.append(f'<span class="metric">Collaboration: <strong>{landscape.get("collaboration_level", "N/A")}</strong></span>')
                html_parts.append('</div>')
                html_parts.append('</div>')
            
            # Trend analysis
            if insights.get("trend_analysis"):
                trends = insights["trend_analysis"]
                html_parts.append('<div class="trend-analysis">')
                html_parts.append('<h4>📈 Research Trends</h4>')
                if trends.get("trend_direction"):
                    html_parts.append(f'<p>Research trend: <strong>{trends["trend_direction"]}</strong></p>')
                if trends.get("peak_year"):
                    html_parts.append(f'<p>Peak research year: <strong>{trends["peak_year"]}</strong></p>')
                html_parts.append('</div>')
            
            html_parts.append('</div>')  # Close enhanced-insights-section
        
        # Enhanced CSS
        html_parts.append('''
        <style>
            .enhanced-chat-response {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1000px;
            }
            
            .enhanced-paper-card {
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                border: 1px solid #e1e4e8;
                border-radius: 12px;
                padding: 1.5em;
                margin: 1em 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: transform 0.2s ease;
            }
            
            .enhanced-paper-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            
            .paper-header {
                display: flex;
                align-items: flex-start;
                gap: 10px;
                margin-bottom: 1em;
            }
            
            .paper-rank {
                background: #0366d6;
                color: white;
                padding: 0.3em 0.6em;
                border-radius: 50%;
                font-weight: bold;
                font-size: 0.9em;
                min-width: 30px;
                text-align: center;
            }
            
            .paper-metadata-enhanced {
                background: #f6f8fa;
                padding: 1em;
                border-radius: 8px;
                margin: 1em 0;
            }
            
            .metadata-row {
                display: flex;
                margin-bottom: 0.5em;
            }
            
            .metadata-label {
                font-weight: 600;
                color: #586069;
                min-width: 80px;
            }
            
            .metadata-value {
                color: #24292e;
            }
            
            .relevance-score {
                background: #28a745;
                color: white;
                padding: 0.2em 0.5em;
                border-radius: 4px;
                font-size: 0.9em;
                font-weight: bold;
            }
            
            .enhanced-categories {
                margin: 1em 0;
            }
            
            .category-tag {
                display: inline-block;
                padding: 0.3em 0.8em;
                margin: 0.2em;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 500;
            }
            
            .category-tag.cs { background: #e1f5fe; color: #01579b; }
            .category-tag.math { background: #f3e5f5; color: #4a148c; }
            .category-tag.physics { background: #e8f5e8; color: #1b5e20; }
            .category-tag.astro { background: #fff3e0; color: #e65100; }
            .category-tag.default { background: #e1ecf4; color: #39739d; }
            
            .abstract-preview {
                background: #fafbfc;
                padding: 1em;
                border-left: 4px solid #0366d6;
                border-radius: 0 8px 8px 0;
                margin-top: 1em;
                font-style: italic;
            }
            
            .enhanced-insights-section {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2em;
                border-radius: 12px;
                margin: 2em 0;
            }
            
            .landscape-summary, .trend-analysis {
                background: rgba(255,255,255,0.1);
                padding: 1em;
                border-radius: 8px;
                margin: 1em 0;
            }
            
            .landscape-metrics {
                display: flex;
                gap: 1em;
                flex-wrap: wrap;
            }
            
            .metric {
                background: rgba(255,255,255,0.2);
                padding: 0.5em 1em;
                border-radius: 20px;
                font-size: 0.9em;
            }
            
            .section-title {
                color: #2c3e50;
                border-bottom: 2px solid #eee;
                padding-bottom: 0.5em;
                margin: 1.5em 0 1em 0;
            }
            
            .enhanced-insights-section .section-title {
                color: white;
                border-bottom: 2px solid rgba(255,255,255,0.3);
            }
        </style>
        ''')
        
        html_parts.append('</div>')  # Close enhanced-chat-response
        return '\n'.join(html_parts)

    def _get_category_class(self, category: str) -> str:
        """
        Get CSS class for category styling
        """
        if category.startswith('cs'):
            return 'cs'
        elif category.startswith('math'):
            return 'math'
        elif category.startswith('physics') or category.startswith('gr-qc') or category.startswith('hep'):
            return 'physics'
        elif category.startswith('astro'):
            return 'astro'
        else:
            return 'default'

    async def _update_session(self, session_id: str, query: str, response_data: Dict[str, Any]):
        """
        Update chat session with query and response
        """
        try:
            with get_db() as db:
                session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
                if session:
                    # Add user message
                    session.messages.append({
                        "role": "user",
                        "content": query,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Add assistant response
                    session.messages.append({
                        "role": "assistant",
                        "content": response_data.get("response", ""),
                        "html_content": response_data.get("html_response", ""),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    db.commit()
        except Exception as e:
            self.logger.error(f"Error updating session: {str(e)}")

    def _generate_error_response(self, query: str, session_id: Optional[str]) -> Dict[str, Any]:
        """
        Generate error response
        """
        return {
            "response": "I apologize, but I encountered an error processing your request. Please try again with a different query.",
            "html_response": '<div class="error-message">I apologize, but I encountered an error processing your request. Please try again with a different query.</div>',
            "arxiv": [],
            "intent": {"type": "error", "needs_search": False},
            "session_id": session_id,
            "metadata": {"error": True}
        } 