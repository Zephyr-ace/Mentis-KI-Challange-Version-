#!/usr/bin/env python3

from core.chat import Chat
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

def test_retriever():
    """Test the retriever with get_connected_objects functionality"""
    
    # Check if required environment variables are set
    if not os.getenv("USER_ID"):
        print("❌ USER_ID environment variable is required")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable is required")  
        return
    
    print("🚀 Testing chat system with connected objects...")
    
    # Test query
    test_query = "Tell me about people and their emotions"
    
    try:
        # Initialize chat system
        with Chat() as chat:
            print(f"\n📝 Query: {test_query}")
            print("🔍 Searching across all retrievers...")
            
            # Get responses from all 3 retrievers
            responses = chat.chat(test_query)
            
            # Display results
            print("\n" + "="*80)
            print("📊 RESULTS FROM 3 DIFFERENT RETRIEVERS")
            print("="*80)
            
            print(f"\n🎯 MAIN RETRIEVER (Semantic Search):")
            print("-" * 50)
            print(responses["main_retriever"])
            
            print(f"\n📝 SIMPLE RAG (Text Chunks):")
            print("-" * 50)
            print(responses["simple_rag"])
            
            print(f"\n📋 SUMMARY RAG (Summarized Content):")
            print("-" * 50)
            print(responses["summary_rag"])
            
            print("\n" + "="*80)
            print("✅ Test completed successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retriever()