#!/usr/bin/env python3
"""
Simple script to run local embeddings from the embedding module.

This script demonstrates how to use the embedding module from the project root.
"""

import sys
import os

# Add the current directory to Python path so we can import the embedding module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding import test_embeddings, update_embeddings

def main():
    print("🏠 Local Embeddings with 1536 Dimensions")
    print("=" * 50)
    
    # Test the embedding setup first
    print("1. Testing embedding setup...")
    try:
        if test_embeddings():
            print("✅ Embedding test passed!")
            
            # Ask user if they want to proceed with full processing
            print("\n2. Ready to process database records...")
            choice = input("Do you want to start processing all records? (y/N): ").lower().strip()
            
            if choice in ['y', 'yes']:
                print("\n🚀 Starting batch processing...")
                print("This will take 30-60 minutes depending on your hardware.")
                print("Processing 500,000 records with 1536-dimensional embeddings...")
                
                update_embeddings()
                
                print("\n✅ Local embedding process completed successfully!")
                print("📊 All records now have 1536-dimensional embeddings.")
            else:
                print("👋 Processing cancelled. You can run this script again anytime.")
                
        else:
            print("❌ Embedding test failed!")
            print("Please check:")
            print("- Dependencies are installed: pip install -r embedding/requirements.txt")
            print("- Database connection is working")
            print("- GPU drivers (if using CUDA)")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Install dependencies: pip install -r embedding/requirements.txt")
        print("2. Check database connection settings")
        print("3. Ensure sufficient disk space for model downloads")
        print("4. Try reducing batch size if memory issues occur")

if __name__ == "__main__":
    main() 