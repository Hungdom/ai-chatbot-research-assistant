from typing import Dict, Any, List
from .base_agent import BaseAgent
import openai
from datetime import datetime
from config.prompts import RESPONSE_AGENT_PROMPTS, EMBEDDING_PROMPTS

class ResponseAgent(BaseAgent):
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the search results and generate a response
        """
        self.logger.info("Processing search results for response generation")
        arxiv = input_data.get("arxiv", [])
        query = input_data.get("query", "")
        search_context = input_data.get("search_context", {})
        
        self.logger.info(f"Processing {len(arxiv)} arxiv papers for query: {query}")
        
        if not arxiv:
            self.logger.info("No arxiv papers found, generating appropriate response")
            return await self._generate_no_results_response(query)
        
        # Analyze arxiv papers based on query intent
        analysis = await self._analyze_arxiv(arxiv, query)
        
        # Check if this is a specific analysis type
        analysis_type = self._determine_analysis_type(query)
        
        # Generate response with analysis
        response = await self._generate_response(arxiv, query, analysis, analysis_type)
        
        return {
            "response": response,
            "arxiv": arxiv,
            "analysis": analysis
        }
    
    def _determine_analysis_type(self, query: str) -> str:
        """
        Determine the type of analysis requested
        """
        query = query.lower()
        if "trend" in query or "popular" in query:
            return "trending_topics"
        elif "year" in query or "this year" in query:
            return "yearly_summary"
        elif "category" in query or "field" in query:
            return "category_analysis"
        return "analysis"
    
    async def _analyze_arxiv(self, arxiv: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Analyze the arxiv papers to extract insights
        """
        # Extract years and categories
        years = [p.get("update_date").year for p in arxiv if p.get("update_date")]
        categories = []
        for item in arxiv:
            if item.get("categories"):
                categories.extend(item["categories"])
        
        # Count category frequencies
        category_freq = {}
        for category in categories:
            category_freq[category] = category_freq.get(category, 0) + 1
        
        # Get most common categories
        top_categories = sorted(category_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Analyze year distribution
        year_distribution = {}
        for year in years:
            year_distribution[year] = year_distribution.get(year, 0) + 1
        
        # Get trending years
        trending_years = sorted(year_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Check for embeddings
        has_embeddings = any(p.get("embedding") is not None for p in arxiv)
        
        return {
            "total_arxiv": len(arxiv),
            "year_range": {
                "min": min(years) if years else None,
                "max": max(years) if years else None
            },
            "top_categories": [{"category": cat, "count": count} for cat, count in top_categories],
            "trending_years": [{"year": year, "count": count} for year, count in trending_years],
            "has_embeddings": has_embeddings
        }
    
    async def _generate_response(self, arxiv: List[Dict[str, Any]], query: str, analysis: Dict[str, Any], analysis_type: str) -> str:
        """
        Generate a response with analysis
        """
        # Prepare context for the AI
        context = {
            "query": query,
            "total_arxiv": analysis['total_arxiv'],
            "year_range": f"{analysis['year_range']['min']} - {analysis['year_range']['max']}",
            "top_categories": ', '.join([f"{cat['category']} ({cat['count']})" for cat in analysis['top_categories']]),
            "trending_years": ', '.join([f"{year['year']} ({year['count']})" for year in analysis['trending_years']]),
            "arxiv": arxiv,
            "categories": analysis['top_categories']
        }
        
        # Select appropriate prompt template
        if analysis_type == "trending_topics":
            prompt = RESPONSE_AGENT_PROMPTS["trending_topics"]
        elif analysis_type == "yearly_summary":
            prompt = RESPONSE_AGENT_PROMPTS["yearly_summary"]
            context["year"] = datetime.now().year
        elif analysis_type == "category_analysis":
            prompt = RESPONSE_AGENT_PROMPTS["category_analysis"]
        else:
            prompt = RESPONSE_AGENT_PROMPTS["analysis"]
        
        # Add embedding analysis if available
        if analysis.get("has_embeddings"):
            embedding_analysis = await self._analyze_embeddings(arxiv, query)
            context["embedding_analysis"] = embedding_analysis
        
        # Generate completion with analysis
        self.logger.info("Generating completion with temperature 0.7")
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": RESPONSE_AGENT_PROMPTS["system"]},
                {"role": "user", "content": prompt.format(**context)}
            ],
            temperature=0.7
        )
        
        self.logger.info("Successfully generated completion")
        return completion.choices[0].message.content
    
    async def _analyze_embeddings(self, arxiv: List[Dict[str, Any]], query: str) -> str:
        """
        Analyze semantic similarities between arxiv papers using embeddings
        """
        # Filter arxiv papers with embeddings
        arxiv_with_embeddings = [p for p in arxiv if p.get("embedding") is not None]
        
        if not arxiv_with_embeddings:
            return ""
        
        # Generate embedding analysis
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": EMBEDDING_PROMPTS["system"]},
                {"role": "user", "content": EMBEDDING_PROMPTS["similarity_analysis"].format(
                    query=query,
                    arxiv=arxiv_with_embeddings
                )}
            ],
            temperature=0.7
        )
        
        return completion.choices[0].message.content
    
    async def _generate_no_results_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response when no arxiv papers are found
        """
        self.logger.info("Generating completion with temperature 0.7")
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": RESPONSE_AGENT_PROMPTS["system"]},
                {"role": "user", "content": RESPONSE_AGENT_PROMPTS["no_results"].format(query=query)}
            ],
            temperature=0.7
        )
        
        self.logger.info("Successfully generated completion")
        return {
            "response": completion.choices[0].message.content,
            "arxiv": [],
            "analysis": {
                "total_arxiv": 0,
                "year_range": {"min": None, "max": None},
                "top_categories": [],
                "trending_years": [],
                "has_embeddings": False
            }
        } 