"""
Configuration file for all agent prompts
"""

# Request Agent Prompts
REQUEST_AGENT_PROMPTS = {
    "system": """You are a research assistant that helps users find and analyze academic papers from ArXiv. 
    Your role is to understand user queries and generate appropriate search parameters.""",
    
    "query_analysis": """Analyze the following query and extract key information:
    - Main topic or subject
    - Time period or year
    - Specific keywords or concepts
    - Type of analysis requested (trending, yearly, category, etc.)
    
    Query: {query}
    
    Provide your analysis in a structured format.""",
    
    "follow_up": """Based on the user's query, generate 3 relevant follow-up questions that could help refine their search:
    Query: {query}
    
    Generate questions that:
    1. Help narrow down the topic
    2. Suggest specific time periods
    3. Recommend related keywords or concepts"""
}

# Query Agent Prompts
QUERY_AGENT_PROMPTS = {
    "system": """You are a research database expert that helps find relevant academic papers from ArXiv.
    You understand how to use both semantic search (embeddings) and traditional text search effectively.""",
    
    "search_strategy": """Based on the query analysis, determine the best search strategy:
    - Use semantic search (embeddings) for conceptual queries
    - Use text search for specific terms or exact matches
    - Combine both approaches for comprehensive results
    
    Query Analysis: {analysis}
    
    Provide your search strategy in a structured format."""
}

# Response Agent Prompts
RESPONSE_AGENT_PROMPTS = {
    "system": """You are a research analyst that provides insightful analysis of academic papers from ArXiv.
    You help users understand trends, patterns, and key findings in the research landscape.""",
    
    "analysis": """Analyze the following ArXiv papers and provide insights:
    
    Query: {query}
    Total Papers: {total_arxiv}
    Year Range: {year_range}
    Top Categories: {top_categories}
    Trending Years: {trending_years}
    
    Focus on:
    1. Key trends and patterns
    2. Notable findings or breakthroughs
    3. Gaps in the research
    4. Future research directions
    
    Provide a comprehensive analysis that helps users understand the research landscape.""",
    
    "no_results": """No ArXiv papers were found for the query: {query}
    
    Provide helpful suggestions for:
    1. Refining the search query
    2. Alternative search strategies
    3. Related topics to explore
    4. Specific ArXiv categories to try
    
    Be specific and actionable in your suggestions.""",
    
    "trending_topics": """Analyze the following ArXiv papers to identify trending topics:
    
    Papers: {arxiv}
    
    Focus on:
    1. Emerging research areas
    2. Popular methodologies
    3. Common applications
    4. Cross-disciplinary connections
    
    Provide insights about current trends in the field.""",
    
    "yearly_summary": """Provide a summary of ArXiv papers from {year}:
    
    Papers: {arxiv}
    
    Include:
    1. Major themes and topics
    2. Key findings and breakthroughs
    3. Notable authors and institutions
    4. Impact and significance
    
    Create a comprehensive overview of research developments in {year}.""",
    
    "category_analysis": """Analyze ArXiv papers by category:
    
    Papers: {arxiv}
    Categories: {categories}
    
    Focus on:
    1. Distribution across categories
    2. Interdisciplinary connections
    3. Category-specific trends
    4. Emerging subcategories
    
    Provide insights about the research landscape across different categories."""
}

# Embedding-related Prompts
EMBEDDING_PROMPTS = {
    "system": """You are an expert in semantic search and paper analysis.
    You understand how to use embeddings to find conceptually similar papers from ArXiv.""",
    
    "similarity_analysis": """Analyze the semantic similarity between ArXiv papers:
    
    Query: {query}
    Papers: {arxiv}
    
    Focus on:
    1. Conceptual similarities
    2. Methodological connections
    3. Thematic relationships
    4. Complementary findings
    
    Provide insights about how these papers are related semantically."""
}

CHAT_AGENT_PROMPTS = {
    "system": """You are a research assistant helping users find and understand academic papers. 
    
Guidelines:
- Be helpful and informative
- Focus on the user's specific query
- Highlight key findings and insights
- Suggest follow-up questions when appropriate
- Keep responses clear and well-structured""",
    
    "paper_search_with_results": """Query: {query}
Search Results: {papers}
{insights}

Based on the papers found, provide:
1. A summary of key findings
2. How these papers relate to the query
3. Important trends or patterns
4. Suggestions for further research

Be concise but comprehensive.""",
    
    "paper_search_no_results": """Query: {query}
No specific papers found matching this query.

Please:
1. Suggest alternative search terms or approaches
2. Recommend related research areas to explore
3. Provide general guidance on this research topic
4. Ask clarifying questions to better understand the user's needs""",
    
    "general_query": """Query: {query}
Context: {context}

Please provide a helpful response addressing the user's question.
If this appears to be a research query, suggest ways to find relevant papers or research directions.""",
    
    "follow_up": """Generate 2-3 helpful follow-up questions for: {query}

Focus on:
- Specific aspects to explore deeper
- Related research areas
- Practical applications

One question per line.""",
    
    "intent_analysis": """Analyze this research query and return JSON with:
- type: paper_search, trend_analysis, author_search, general_question
- categories: list of relevant research categories
- search_terms: key terms to search for
- time_scope: recent, historical, all_time
- needs_search: true/false

Query: {query}

Respond with valid JSON only."""
} 