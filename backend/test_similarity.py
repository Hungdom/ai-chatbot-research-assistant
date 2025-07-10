#!/usr/bin/env python3
"""
Quick test for optimized similarity calculation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from agents.smart_search_agent import SmartSearchAgent
from datetime import datetime

# Mock paper class for testing
class MockPaper:
    def __init__(self, title, abstract, categories, authors=None, published_date=None):
        self.title = title
        self.abstract = abstract
        self.categories = categories
        self.authors = authors or ["Test Author"]
        self.published_date = published_date or datetime.now()

def test_similarity():
    agent = SmartSearchAgent()
    
    # Test papers
    papers = [
        MockPaper(
            "Machine Learning for Data Analysis",
            "This paper presents a comprehensive study of machine learning techniques for data analysis and pattern recognition in large datasets.",
            ["cs.LG", "stat.ML"]
        ),
        MockPaper(
            "Deep Learning Neural Networks",
            "We propose a new deep learning architecture using neural networks for image classification tasks.",
            ["cs.CV", "cs.LG"]
        ),
        MockPaper(
            "Quantum Computing Algorithms",
            "This work explores quantum algorithms for solving optimization problems in quantum computing systems.",
            ["quant-ph", "cs.DS"]
        ),
        MockPaper(
            "Differential Equations in Physics",
            "Mathematical analysis of differential equations and their applications in theoretical physics.",
            ["math.AP", "physics.math-ph"]
        ),
        MockPaper(
            "Statistical Learning Theory",
            "An introduction to statistical learning theory with applications to machine learning and data science.",
            ["stat.ML", "cs.LG"]
        )
    ]
    
    # Test queries
    test_queries = [
        "machine learning",
        "deep learning neural networks",
        "quantum computing",
        "differential equations",
        "data analysis",
        "statistical learning",
        "computer vision",
        "optimization algorithms"
    ]
    
    print("Testing Optimized Similarity Calculation")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 30)
        
        # Calculate similarity for each paper
        results = []
        for paper in papers:
            similarity = agent._calculate_keyword_similarity(query, paper)
            results.append((paper.title, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        for title, similarity in results:
            if similarity > 0.0:
                print(f"  {similarity:.2f} ({similarity:.1%}) - {title}")
            else:
                print(f"  {similarity:.2f} ({similarity:.1%}) - {title} (filtered out)")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_similarity() 