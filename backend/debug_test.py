#!/usr/bin/env python3
"""
Debug test to identify the source of the error
"""

import sys
import os
import asyncio
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(__file__))

async def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    try:
        from agents.enhanced_chat_agent import EnhancedChatAgent
        print("✓ Enhanced chat agent imported successfully")
        
        from agents.smart_search_agent import SmartSearchAgent
        print("✓ Smart search agent imported successfully")
        
        from agents.chat_agent import ChatAgent
        print("✓ Chat agent imported successfully")
        
        from database import get_db, Arxiv
        print("✓ Database imports successful")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {str(e)}")
        return False

async def test_database_connection():
    """Test database connection"""
    print("\nTesting database connection...")
    try:
        from database import get_db, Arxiv
        from sqlalchemy import text
        
        with get_db() as db:
            result = db.execute(text("SELECT COUNT(*) FROM arxiv LIMIT 1")).scalar()
            print(f"✓ Database connected - {result} papers found")
            return True
    except Exception as e:
        print(f"✗ Database error: {str(e)}")
        return False

async def test_agent_creation():
    """Test agent creation"""
    print("\nTesting agent creation...")
    try:
        from agents.enhanced_chat_agent import EnhancedChatAgent
        from agents.smart_search_agent import SmartSearchAgent
        
        enhanced_agent = EnhancedChatAgent()
        print("✓ Enhanced chat agent created")
        
        smart_agent = SmartSearchAgent()
        print("✓ Smart search agent created")
        
        return True
    except Exception as e:
        print(f"✗ Agent creation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_simple_query():
    """Test a simple query processing"""
    print("\nTesting simple query processing...")
    try:
        from agents.enhanced_chat_agent import EnhancedChatAgent
        
        agent = EnhancedChatAgent()
        
        # Test with a very simple query
        result = await agent.process({
            "query": "test",
            "context": {},
            "session_id": None
        })
        
        print(f"✓ Simple query processed - Response length: {len(result.get('response', ''))}")
        return True
    except Exception as e:
        print(f"✗ Query processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_similarity_calculation():
    """Test the optimized similarity calculation"""
    print("\nTesting similarity calculation...")
    try:
        from agents.smart_search_agent import SmartSearchAgent
        
        # Mock paper for testing
        class MockPaper:
            def __init__(self):
                self.title = "Test Paper About Machine Learning"
                self.abstract = "This is a test abstract about machine learning and data analysis."
                self.categories = ["cs.LG"]
                self.authors = ["Test Author"]
                self.published_date = datetime.now()
        
        agent = SmartSearchAgent()
        paper = MockPaper()
        
        similarity = agent._calculate_keyword_similarity("machine learning", paper)
        print(f"✓ Similarity calculation works - Score: {similarity:.3f}")
        return True
    except Exception as e:
        print(f"✗ Similarity calculation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("Starting debug tests...")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_database_connection,
        test_agent_creation,
        test_similarity_calculation,
        test_simple_query
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed < total:
        print("\nSome tests failed. Check the errors above to identify the issue.")
    else:
        print("\nAll tests passed! The issue might be in the API configuration or specific input.")

if __name__ == "__main__":
    asyncio.run(main()) 