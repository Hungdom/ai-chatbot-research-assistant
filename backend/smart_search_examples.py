#!/usr/bin/env python3
"""
Examples demonstrating the enhanced smart search capabilities
"""

import asyncio
from agents.enhanced_chat_agent import EnhancedChatAgent

async def example_category_hierarchy_search():
    """
    Example: Leveraging category hierarchy for broader search
    """
    agent = EnhancedChatAgent()
    
    # Query about machine learning - should expand to related categories
    query = "machine learning applications in astrophysics"
    
    result = await agent.process({
        "query": query,
        "context": {}
    })
    
    print("=== Category Hierarchy Search ===")
    print(f"Query: {query}")
    print(f"Categories found: {result['intent'].get('categories', [])}")
    print(f"Expanded categories: {result['intent'].get('expanded_categories', [])}")
    print(f"Papers found: {len(result['arxiv'])}")
    
    # Show how it found papers across multiple related categories
    categories_in_results = set()
    for paper in result['arxiv']:
        categories_in_results.update(paper.get('categories', []))
    
    print(f"Actual categories in results: {list(categories_in_results)[:10]}")
    print()

async def example_temporal_trend_analysis():
    """
    Example: Temporal analysis and trend detection
    """
    agent = EnhancedChatAgent()
    
    # Query about research trends over time
    query = "evolution of deep learning research in computer vision from 2015 to 2023"
    
    result = await agent.process({
        "query": query,
        "context": {}
    })
    
    print("=== Temporal Trend Analysis ===")
    print(f"Query: {query}")
    print(f"Temporal scope: {result['intent'].get('temporal_scope', 'N/A')}")
    print(f"Years detected: {result['intent'].get('temporal_filters', {}).get('years', [])}")
    
    if result['intent'].get('insights', {}).get('trend_analysis'):
        trends = result['intent']['insights']['trend_analysis']
        print(f"Peak year: {trends.get('peak_year', 'N/A')}")
        print(f"Trend direction: {trends.get('trend_direction', 'N/A')}")
        print(f"Papers by year: {trends.get('paper_count_by_year', {})}")
    
    print()

async def example_interdisciplinary_search():
    """
    Example: Finding interdisciplinary research
    """
    agent = EnhancedChatAgent()
    
    # Query about interdisciplinary work
    query = "interdisciplinary research combining quantum physics and machine learning"
    
    result = await agent.process({
        "query": query,
        "context": {}
    })
    
    print("=== Interdisciplinary Search ===")
    print(f"Query: {query}")
    print(f"Interdisciplinary flag: {result['intent'].get('interdisciplinary', False)}")
    print(f"Research areas: {result['intent'].get('research_areas', [])}")
    
    # Show papers with multiple categories
    interdisciplinary_papers = [
        p for p in result['arxiv'] 
        if len(p.get('categories', [])) > 2
    ]
    
    print(f"Papers with 3+ categories: {len(interdisciplinary_papers)}")
    
    if interdisciplinary_papers:
        example_paper = interdisciplinary_papers[0]
        print(f"Example: {example_paper['title'][:60]}...")
        print(f"Categories: {example_paper['categories']}")
    
    print()

async def example_author_collaboration_analysis():
    """
    Example: Author-based search and collaboration analysis
    """
    agent = EnhancedChatAgent()
    
    # Query about specific researcher's work
    query = "Geoffrey Hinton's recent work on neural networks and deep learning"
    
    result = await agent.process({
        "query": query,
        "context": {}
    })
    
    print("=== Author Collaboration Analysis ===")
    print(f"Query: {query}")
    print(f"Author filters: {result['intent'].get('author_filters', [])}")
    
    if result['intent'].get('insights', {}).get('author_analysis'):
        author_analysis = result['intent']['insights']['author_analysis']
        print(f"Top authors: {list(author_analysis.get('top_authors', {}).keys())[:5]}")
        print(f"Top collaborators: {list(author_analysis.get('top_collaborators', {}).keys())[:3]}")
        print(f"Unique authors: {author_analysis.get('total_unique_authors', 0)}")
    
    print()

async def example_methodology_based_search():
    """
    Example: Searching by research methodology
    """
    agent = EnhancedChatAgent()
    
    # Query about specific methodologies
    query = "experimental studies on quantum computing using superconducting qubits"
    
    result = await agent.process({
        "query": query,
        "context": {}
    })
    
    print("=== Methodology-Based Search ===")
    print(f"Query: {query}")
    print(f"Methodologies detected: {result['intent'].get('methodologies', [])}")
    print(f"Research level: {result['intent'].get('research_level', 'all')}")
    
    # Show papers categorized by methodology
    experimental_papers = [
        p for p in result['arxiv']
        if any(word in (p.get('title', '') + p.get('abstract', '')).lower() 
               for word in ['experiment', 'measurement', 'observation'])
    ]
    
    print(f"Papers with experimental keywords: {len(experimental_papers)}")
    print()

async def example_research_landscape_analysis():
    """
    Example: Comprehensive research landscape analysis
    """
    agent = EnhancedChatAgent()
    
    # Query for landscape analysis
    query = "current state of quantum machine learning research"
    
    result = await agent.process({
        "query": query,
        "context": {}
    })
    
    print("=== Research Landscape Analysis ===")
    print(f"Query: {query}")
    
    if result['intent'].get('insights', {}).get('research_landscape'):
        landscape = result['intent']['insights']['research_landscape']
        print(f"Research maturity: {landscape.get('research_maturity', 'N/A')}")
        print(f"Research intensity: {landscape.get('research_intensity', 0):.2f} papers/year")
        print(f"Interdisciplinary index: {landscape.get('interdisciplinary_index', 0):.2f}")
        print(f"Collaboration level: {landscape.get('collaboration_level', 'N/A')}")
        print(f"Research activity: {landscape.get('research_activity', 'N/A')}")
    
    if result['intent'].get('insights', {}).get('category_analysis'):
        cat_analysis = result['intent']['insights']['category_analysis']
        print(f"Category diversity: {cat_analysis.get('category_diversity', 0)}")
        print(f"Avg categories per paper: {cat_analysis.get('average_categories_per_paper', 0):.2f}")
        print(f"Interdisciplinary ratio: {cat_analysis.get('interdisciplinary_ratio', 0):.2f}")
    
    print()

async def example_smart_query_understanding():
    """
    Example: Demonstrating intelligent query parsing
    """
    test_queries = [
        "recent advances in transformer architectures for NLP",
        "foundational papers on general relativity and black holes",
        "collaboration patterns in computational biology research",
        "emerging trends in quantum error correction",
        "comparison between CNN and Vision Transformer approaches"
    ]
    
    agent = EnhancedChatAgent()
    
    print("=== Smart Query Understanding ===")
    
    for query in test_queries:
        result = await agent.process({
            "query": query,
            "context": {}
        })
        
        intent = result['intent']
        
        print(f"\nQuery: {query}")
        print(f"  Type: {intent.get('query_type', 'N/A')}")
        print(f"  Research level: {intent.get('research_level', 'N/A')}")
        print(f"  Temporal scope: {intent.get('temporal_scope', 'N/A')}")
        print(f"  Complexity: {intent.get('complexity_level', 'N/A')}")
        print(f"  Strategy: {intent.get('search_strategy', 'N/A')}")
        print(f"  Papers found: {len(result['arxiv'])}")

async def run_all_examples():
    """
    Run all examples to demonstrate capabilities
    """
    print("🚀 Enhanced ArXiv Search System - Capability Demonstrations\n")
    
    try:
        await example_category_hierarchy_search()
        await example_temporal_trend_analysis()
        await example_interdisciplinary_search()
        await example_author_collaboration_analysis()
        await example_methodology_based_search()
        await example_research_landscape_analysis()
        await example_smart_query_understanding()
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")

if __name__ == "__main__":
    # Run examples
    asyncio.run(run_all_examples()) 