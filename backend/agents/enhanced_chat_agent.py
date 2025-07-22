#!/usr/bin/env python3
"""
Enhanced Chat Agent - Integrates smart search capabilities
"""

from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from .smart_search_agent import SmartSearchAgent
from database import get_db, Arxiv, ChatSession
from sqlalchemy import text, func, desc, or_, extract, select
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

    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation: 1 token ≈ 4 characters)
        """
        return len(text) // 4

    def _truncate_papers_for_context(self, papers: List[Dict[str, Any]], max_tokens: int = 6000) -> List[Dict[str, Any]]:
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
                "abstract": paper["abstract"][:250] + "..." if len(paper["abstract"]) > 250 else paper["abstract"],
                "authors": paper["authors"][:2],  # Only first 2 authors
                "categories": paper["categories"][:3],  # Only first 3 categories
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
        
        self.logger.info(f"Enhanced agent - Truncated papers: {len(papers)} -> {len(truncated_papers)} (estimated {current_tokens} tokens)")
        return truncated_papers

    async def _generate_enhanced_response(self, query: str, intent: Dict[str, Any], papers: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate enhanced response with rich insights - TOKEN OPTIMIZED
        """
        # 1. TRUNCATE PAPERS TO PREVENT TOKEN OVERFLOW
        truncated_papers = self._truncate_papers_for_context(papers, max_tokens=5000)
        
        # 2. Prepare COMPACT context
        response_context = {
            "query": query,
            "intent": intent,
            "papers": truncated_papers,
            "context": context,
            "insights": intent.get("insights", {}),
            "metadata": {
                "papers_found": len(papers),
                "search_strategy": intent.get("search_strategy", "basic"),
                "query_complexity": intent.get("complexity_level", "moderate")
            }
        }
        
        # 3. Generate response using GPT-4 with CONCISE prompts
        system_prompt = """You are an advanced research assistant. 

RESPONSE FORMAT REQUIREMENTS:
- Use clear markdown formatting with headers (##, ###)
- Structure responses with numbered or bulleted lists
- Include paper titles in **bold** and authors in *italics*
- Use proper line breaks and spacing for readability
- Organize information in logical sections
- End with actionable next steps

Provide comprehensive, insightful responses that:
1. Address the user's specific query
2. Summarize key findings from the papers
3. Highlight important trends and patterns  
4. Suggest follow-up research directions

Structure your responses as follows:

## 📚 Research Overview
[Brief summary of the research landscape for this query]

## 🔍 Key Findings
[Highlight 2-3 most important discoveries or insights]

## 📊 Research Analysis
[Analyze patterns, trends, methodologies, applications]

## 🎯 Research Recommendations
[Specific suggestions for further investigation]

---
**Next Steps**: [Actionable recommendations for the user]"""
        
        # 4. CREATE COMPACT USER PROMPT
        papers_summary = f"Found {len(papers)} papers"
        if truncated_papers:
            top_similarity = max(p.get('similarity', 0) for p in truncated_papers)
            papers_summary += f" (best match: {top_similarity:.1%})"
        
        # Extract key insights compactly
        insights_summary = ""
        insights = intent.get("insights", {})
        if insights:
            insight_parts = []
            if insights.get("research_landscape", {}).get("research_maturity"):
                insight_parts.append(f"Field maturity: {insights['research_landscape']['research_maturity']}")
            if insights.get("trend_analysis", {}).get("trend_direction"):
                insight_parts.append(f"Trend: {insights['trend_analysis']['trend_direction']}")
            insights_summary = " | ".join(insight_parts)
        
        user_prompt = f"""Query: {query}
        
Results: {papers_summary}
{f"Insights: {insights_summary}" if insights_summary else ""}

Analysis: {json.dumps(intent.get('query_type', 'general'), ensure_ascii=False)}

Provide a comprehensive response addressing the research query."""
        
        try:
            # 5. TOKEN CHECK BEFORE API CALL
            prompt_tokens = self._count_tokens(user_prompt)
            system_tokens = self._count_tokens(system_prompt)
            total_tokens = prompt_tokens + system_tokens
            
            self.logger.info(f"Enhanced agent token usage: {total_tokens} tokens")
            
            # If too long, use minimal prompt
            if total_tokens > 14000:
                self.logger.warning("Enhanced prompt too long, using minimal format")
                user_prompt = f"Query: {query}\nFound {len(papers)} relevant papers.\nProvide insights and recommendations."
            
            completion = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800  # Limit response length for GPT-4
            )
            
            response_text = completion.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced response: {str(e)}")
            response_text = self._generate_fallback_response(query, papers, intent)
        
        # 6. Generate HTML response using ORIGINAL papers list
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
            return f"""## 🔍 **Search Results for: "{query}"**

Unfortunately, I couldn't find specific papers matching your query, but I can help you explore this topic further.

### 📋 **Alternative Approaches**
- Try broader or alternative keywords
- Explore related research areas  
- Search specific arXiv categories
- Look for papers by key researchers in this field

### 💡 **Suggested Next Steps**
1. **Refine your search terms** - Consider synonyms or related concepts
2. **Explore related categories** - Browse relevant arXiv subject areas
3. **Ask for guidance** - I can help you identify better search strategies

---
**How can I help?** Feel free to rephrase your query or ask about specific aspects of this research area."""
        
        response_parts = [
            f"## 📚 **Research Results for: \"{query}\"**",
            "",
            f"I found **{len(papers)} relevant papers** for your query. Here's a summary:",
            "",
            "### 🔍 **Key Research Areas**"
        ]
        
        # Add insights if available
        insights = intent.get("insights", {})
        if insights.get("top_categories"):
            categories = list(insights["top_categories"].keys())[:3]
            response_parts.append(f"- **Primary Categories**: {', '.join(categories)}")
        
        if insights.get("year_range"):
            year_range = insights["year_range"]
            response_parts.append(f"- **Research Timeline**: {year_range.get('min', 'N/A')} to {year_range.get('max', 'N/A')}")
        
        # Add top papers if available
        if papers:
            response_parts.extend([
                "",
                "### 📄 **Notable Papers**"
            ])
            for i, paper in enumerate(papers[:3]):
                title = paper.get('title', 'Untitled')
                authors = paper.get('authors', 'Unknown authors')
                response_parts.append(f"{i+1}. **{title}** by *{authors}*")
        
        response_parts.extend([
            "",
            "### 💡 **Research Insights**",
            "- Research spans multiple years showing sustained interest",
            "- Active development across various methodologies",
            "- Growing applications in practical domains",
            "",
            "---",
            "**Next Steps**: I can help you explore these papers further, analyze specific aspects, or dive deeper into particular research directions."
        ])
        
        return "\n".join(response_parts)

    async def _format_enhanced_html_response(self, response_text: str, papers: List[Dict[str, Any]], intent: Dict[str, Any]) -> str:
        """
        Format enhanced HTML response with rich visualizations - using structured HTML formatting
        """
        import re
        
        html_parts = ['<div class="enhanced-research-response">']
        
        # Convert structured markdown to HTML (same as chat_agent but enhanced)
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
                    html_parts.append(f'<div class="enhanced-section-header"><span class="enhanced-section-emoji">{emoji}</span><h2 class="enhanced-section-title">{title.strip()}</h2></div>')
                else:
                    html_parts.append(f'<h2 class="enhanced-section-title">{header_text}</h2>')
            
            elif line.startswith('### '):
                if in_list:
                    html_parts.append(f'</{list_type}>')
                    in_list = False
                subsection_text = line[4:].strip()
                emoji_match = re.match(r'^([^\w\s]+)\s*\*?\*?(.+?)\*?\*?$', subsection_text)
                if emoji_match:
                    emoji, title = emoji_match.groups()
                    html_parts.append(f'<div class="enhanced-subsection-header"><span class="enhanced-subsection-emoji">{emoji}</span><h3 class="enhanced-subsection-title">{title.strip()}</h3></div>')
                else:
                    html_parts.append(f'<h3 class="enhanced-subsection-title">{subsection_text}</h3>')
            
            # Lists
            elif line.startswith('- ') or line.startswith('• '):
                if not in_list:
                    html_parts.append('<ul class="enhanced-research-list">')
                    in_list = True
                    list_type = 'ul'
                list_content = line[2:].strip()
                # Handle bold/italic formatting
                list_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', list_content)
                list_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', list_content)
                html_parts.append(f'<li class="enhanced-research-item">{list_content}</li>')
            
            elif re.match(r'^\d+\.\s+', line):
                if not in_list:
                    html_parts.append('<ol class="enhanced-research-list numbered">')
                    in_list = True
                    list_type = 'ol'
                list_content = re.sub(r'^\d+\.\s+', '', line)
                list_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', list_content)
                list_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', list_content)
                html_parts.append(f'<li class="enhanced-research-item">{list_content}</li>')
            
            # Horizontal rule
            elif line.startswith('---'):
                if in_list:
                    html_parts.append(f'</{list_type}>')
                    in_list = False
                html_parts.append('<hr class="enhanced-section-divider"/>')
            
            # Next Steps highlight
            elif line.startswith('**Next Steps'):
                if in_list:
                    html_parts.append(f'</{list_type}>')
                    in_list = False
                next_steps_text = re.sub(r'\*\*(.+?)\*\*:', r'<strong>\1</strong>:', line)
                html_parts.append(f'<div class="enhanced-next-steps-section">{next_steps_text}</div>')
            
            # Regular paragraphs
            else:
                if in_list:
                    html_parts.append(f'</{list_type}>')
                    in_list = False
                # Handle bold/italic formatting
                formatted_line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
                formatted_line = re.sub(r'\*(.+?)\*', r'<em>\1</em>', formatted_line)
                html_parts.append(f'<p class="enhanced-research-text">{formatted_line}</p>')
        
        # Close any open lists
        if in_list:
            html_parts.append(f'</{list_type}>')
        
        # Enhanced papers section with richer metadata
        if papers:
            html_parts.append('<div class="enhanced-papers-showcase">')
            html_parts.append('<div class="enhanced-section-header"><span class="enhanced-section-emoji">📚</span><h2 class="enhanced-section-title">Research Papers Found</h2></div>')
            
            for i, paper in enumerate(papers[:6]):  # Show top 6 papers for enhanced version
                html_parts.append('<div class="premium-paper-card">')
                
                # Paper header with enhanced ranking
                html_parts.append(f'<div class="premium-paper-header">')
                html_parts.append(f'<span class="premium-paper-rank">#{i+1}</span>')
                html_parts.append(f'<div class="premium-paper-info">')
                html_parts.append(f'<h4 class="premium-paper-title"><a href="https://arxiv.org/abs/{paper["arxiv_id"]}" target="_blank">{paper["title"]}</a></h4>')
                
                # Enhanced metadata row
                html_parts.append('<div class="premium-metadata-row">')
                authors = paper.get("authors", [])
                authors_display = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_display += " et al."
                html_parts.append(f'<span class="premium-authors">👥 {authors_display}</span>')
                
                if paper.get("published_date"):
                    html_parts.append(f'<span class="premium-date">📅 {paper["published_date"][:10]}</span>')
                
                if paper.get("similarity", 0) > 0:
                    relevance = paper["similarity"] * 100
                    html_parts.append(f'<span class="premium-relevance">🎯 {relevance:.0f}% match</span>')
                
                html_parts.append('</div>')
                html_parts.append('</div>')
                html_parts.append('</div>')
                
                # Premium categories with enhanced styling
                categories = paper.get("categories", [])[:4]
                if categories:
                    html_parts.append('<div class="premium-categories">')
                    for cat in categories:
                        category_class = self._get_category_class(cat)
                        html_parts.append(f'<span class="premium-category-tag {category_class}">{cat}</span>')
                    html_parts.append('</div>')
                
                # Abstract preview with gradient fade
                abstract = paper.get("abstract", "")
                if abstract:
                    preview = abstract[:180] + "..." if len(abstract) > 180 else abstract
                    html_parts.append('<div class="premium-abstract-preview">')
                    html_parts.append(f'<p>{preview}</p>')
                    html_parts.append('<div class="abstract-fade"></div>')
                    html_parts.append('</div>')
                
                html_parts.append('</div>')  # Close premium-paper-card
            
            html_parts.append('</div>')  # Close enhanced-papers-showcase
        
        # Enhanced insights with premium styling
        insights = intent.get("insights", {})
        if insights:
            html_parts.append('<div class="premium-insights-section">')
            html_parts.append('<div class="enhanced-section-header"><span class="enhanced-section-emoji">📊</span><h2 class="enhanced-section-title">Advanced Research Insights</h2></div>')
            
            # Research landscape visualization
            if insights.get("research_landscape"):
                landscape = insights["research_landscape"]
                html_parts.append('<div class="landscape-visualization">')
                html_parts.append('<h4>🌍 Research Landscape Analysis</h4>')
                html_parts.append(f'<div class="insight-metrics">')
                html_parts.append(f'<div class="metric-card maturity"><span class="metric-label">Field Maturity</span><span class="metric-value">{landscape.get("research_maturity", "Emerging")}</span></div>')
                html_parts.append(f'<div class="metric-card activity"><span class="metric-label">Activity Level</span><span class="metric-value">{landscape.get("research_activity", "Active")}</span></div>')
                html_parts.append(f'<div class="metric-card collaboration"><span class="metric-label">Collaboration</span><span class="metric-value">{landscape.get("collaboration_level", "High")}</span></div>')
                html_parts.append('</div>')
                html_parts.append('</div>')
            
            # Trend analysis with visualization
            if insights.get("trend_analysis"):
                trends = insights["trend_analysis"]
                html_parts.append('<div class="trend-visualization">')
                html_parts.append('<h4>📈 Research Trend Analysis</h4>')
                if trends.get("trend_direction"):
                    html_parts.append(f'<div class="trend-indicator {trends["trend_direction"].lower()}">')
                    html_parts.append(f'<span class="trend-arrow"></span>')
                    html_parts.append(f'<span class="trend-text">Research trend: <strong>{trends["trend_direction"]}</strong></span>')
                    html_parts.append('</div>')
                if trends.get("peak_year"):
                    html_parts.append(f'<p class="peak-info">📊 Peak activity: <strong>{trends["peak_year"]}</strong></p>')
                html_parts.append('</div>')
            
            html_parts.append('</div>')  # Close premium-insights-section
        
        # Premium CSS styles
        html_parts.append('''
        <style>
            .enhanced-research-response {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                line-height: 1.7;
                color: #1a202c;
                max-width: 100%;
                padding: 1.5rem 0;
            }
            
            .enhanced-section-header, .enhanced-subsection-header {
                display: flex;
                align-items: center;
                margin: 2rem 0 1.25rem 0;
                gap: 0.75rem;
            }
            
            .enhanced-section-emoji, .enhanced-subsection-emoji {
                font-size: 1.5em;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
            }
            
            .enhanced-section-title {
                font-size: 1.375rem;
                font-weight: 800;
                color: #1a202c;
                margin: 0;
                border-bottom: 3px solid #4f46e5;
                padding-bottom: 0.75rem;
                flex-grow: 1;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .enhanced-subsection-title {
                font-size: 1.125rem;
                font-weight: 700;
                color: #374151;
                margin: 0;
            }
            
            .enhanced-research-list {
                margin: 1.25rem 0;
                padding-left: 2rem;
            }
            
            .enhanced-research-item {
                margin: 1rem 0;
                padding: 0.75rem 0;
                line-height: 1.7;
                position: relative;
            }
            
            .enhanced-research-item strong {
                color: #3730a3;
                font-weight: 700;
            }
            
            .enhanced-research-text {
                margin: 1.25rem 0;
                line-height: 1.8;
                font-size: 1.05rem;
            }
            
            .enhanced-section-divider {
                border: none;
                height: 3px;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
                margin: 2.5rem 0;
                border-radius: 1.5px;
            }
            
            .enhanced-next-steps-section {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 16px;
                margin: 2rem 0;
                box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
                position: relative;
                overflow: hidden;
            }
            
            .enhanced-next-steps-section::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #fbbf24, #f59e0b, #fbbf24);
            }
            
            .enhanced-papers-showcase {
                margin-top: 2.5rem;
                padding: 2rem;
                background: linear-gradient(135deg, #f8faff 0%, #e0e7ff 100%);
                border-radius: 20px;
                border: 2px solid #c7d2fe;
                box-shadow: 0 4px 16px rgba(99, 102, 241, 0.1);
            }
            
            .premium-paper-card {
                background: white;
                padding: 1.75rem;
                margin: 1.5rem 0;
                border-radius: 16px;
                border: 2px solid #e5e7eb;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .premium-paper-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #3b82f6, #1d4ed8, #3b82f6);
                transform: translateX(-100%);
                transition: transform 0.3s ease;
            }
            
            .premium-paper-card:hover::before {
                transform: translateX(0);
            }
            
            .premium-paper-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 12px 28px rgba(0,0,0,0.15);
                border-color: #3b82f6;
            }
            
            .premium-paper-header {
                display: flex;
                align-items: flex-start;
                gap: 1.25rem;
                margin-bottom: 1.5rem;
            }
            
            .premium-paper-rank {
                background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
                color: white;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 1rem;
                flex-shrink: 0;
                box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
            }
            
            .premium-paper-info {
                flex-grow: 1;
            }
            
            .premium-paper-title {
                margin: 0 0 1rem 0;
                font-size: 1.125rem;
                line-height: 1.5;
            }
            
            .premium-paper-title a {
                color: #1f2937;
                text-decoration: none;
                font-weight: 700;
            }
            
            .premium-paper-title a:hover {
                color: #3b82f6;
            }
            
            .premium-metadata-row {
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                font-size: 0.9rem;
            }
            
            .premium-authors, .premium-date, .premium-relevance {
                background: #f3f4f6;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                color: #374151;
                font-weight: 500;
            }
            
            .premium-relevance {
                background: #dcfce7;
                color: #166534;
            }
            
            .premium-categories {
                display: flex;
                flex-wrap: wrap;
                gap: 0.75rem;
                margin: 1.5rem 0;
            }
            
            .premium-category-tag {
                padding: 0.5rem 1rem;
                border-radius: 25px;
                font-size: 0.85rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.025em;
            }
            
            .premium-category-tag.cs { 
                background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                color: #1e3a8a; 
                border: 1px solid #93c5fd;
            }
            .premium-category-tag.math { 
                background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); 
                color: #701a75; 
                border: 1px solid #f0abfc;
            }
            .premium-category-tag.physics { 
                background: linear-gradient(135deg, #d1fae5 0%, #bbf7d0 100%); 
                color: #14532d; 
                border: 1px solid #86efac;
            }
            .premium-category-tag.default { 
                background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); 
                color: #475569; 
                border: 1px solid #cbd5e1;
            }
            
            .premium-abstract-preview {
                background: #fafbfc;
                padding: 1.5rem;
                border-left: 4px solid #3b82f6;
                border-radius: 0 12px 12px 0;
                margin-top: 1.5rem;
                font-style: italic;
                position: relative;
                overflow: hidden;
            }
            
            .abstract-fade {
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 20px;
                background: linear-gradient(transparent, #fafbfc);
            }
            
            .premium-insights-section {
                margin-top: 3rem;
                padding: 2.5rem;
                background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
                color: white;
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(30, 27, 75, 0.3);
            }
            
            .landscape-visualization, .trend-visualization {
                background: rgba(255,255,255,0.1);
                padding: 2rem;
                border-radius: 12px;
                margin: 2rem 0;
                backdrop-filter: blur(10px);
            }
            
            .insight-metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin: 1.5rem 0;
            }
            
            .metric-card {
                background: rgba(255,255,255,0.15);
                padding: 1.5rem;
                border-radius: 12px;
                text-align: center;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255,255,255,0.2);
            }
            
            .metric-label {
                display: block;
                font-size: 0.9rem;
                opacity: 0.8;
                margin-bottom: 0.5rem;
            }
            
            .metric-value {
                display: block;
                font-size: 1.25rem;
                font-weight: bold;
                color: #fbbf24;
            }
            
            .trend-indicator {
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 1rem;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
                margin: 1rem 0;
            }
            
            .trend-arrow {
                width: 0;
                height: 0;
                border-style: solid;
            }
            
            .trend-indicator.increasing .trend-arrow {
                border-left: 10px solid transparent;
                border-right: 10px solid transparent;
                border-bottom: 15px solid #10b981;
            }
            
            .trend-indicator.stable .trend-arrow {
                border-top: 8px solid transparent;
                border-bottom: 8px solid transparent;
                border-left: 15px solid #f59e0b;
            }
            
            .peak-info {
                color: #fbbf24;
                font-weight: 600;
            }
        </style>
        ''')
        
        html_parts.append('</div>')  # Close enhanced-research-response
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