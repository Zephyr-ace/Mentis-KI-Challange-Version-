#!/usr/bin/env python3
"""
Analyze MainRetriever's specific issues and potential improvements
"""
import warnings
warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.retriever import Retriever
from core.vector_db import VectorDB

def analyze_query_rewriting():
    """Analyze how queries are being rewritten and classified"""
    print("=== ANALYZING QUERY REWRITING ===\n")
    
    test_queries = [
        "What gifts did Anne receive for her birthday?",
        "Tell me about Anne's relationship with her mother",
        "What emotions did Anne express about school?"
    ]
    
    with Retriever() as retriever:
        for query in test_queries:
            print(f"Original: '{query}'")
            result = retriever.retrieve(query)
            
            if "queries_used" in result:
                print("Rewritten queries:")
                for rewritten in result["queries_used"]:
                    print(f"  → '{rewritten['query']}' ({rewritten['category']})")
            print()

def analyze_connection_problems():
    """Analyze the connection system issues"""
    print("=== ANALYZING CONNECTION SYSTEM ===\n")
    
    with VectorDB() as db:
        # Get all connections
        try:
            connections = db.client.collections.get("Connection")
            all_connections = connections.query.fetch_objects(limit=20)
            
            print(f"Found {len(all_connections.objects)} connections in database:")
            
            connection_issues = 0
            for i, conn in enumerate(all_connections.objects):
                source_id = conn.properties.get('source_id')
                target_id = conn.properties.get('target_id')
                conn_type = conn.properties.get('type')
                
                print(f"{i+1}. {source_id} -> {target_id} ({conn_type})")
                
                # Check if target exists in any collection
                target_found = False
                collection_names = ["ChunkEvent", "ChunkPerson", "ChunkEmotion", "ChunkThought", "ChunkProblem", "ChunkAchievement", "ChunkFutureIntention"]
                
                for coll_name in collection_names:
                    try:
                        collection = db.client.collections.get(coll_name)
                        target_obj = collection.query.fetch_objects(
                            where={"path": ["object_id"], "operator": "Equal", "valueText": target_id},
                            limit=1
                        )
                        if target_obj.objects:
                            target_found = True
                            break
                    except:
                        continue
                
                if not target_found:
                    print(f"   ❌ Target {target_id} not found in any collection!")
                    connection_issues += 1
            
            print(f"\n🔍 Connection Analysis:")
            print(f"  Total connections: {len(all_connections.objects)}")
            print(f"  Broken connections: {connection_issues}")
            print(f"  Success rate: {((len(all_connections.objects) - connection_issues) / len(all_connections.objects) * 100):.1f}%")
            
        except Exception as e:
            print(f"Error analyzing connections: {e}")

def analyze_search_distribution():
    """Analyze how search results are distributed across categories"""
    print("\n=== ANALYZING SEARCH DISTRIBUTION ===\n")
    
    test_query = "What emotions did Anne express about her birthday?"
    
    with Retriever() as retriever:
        result = retriever.retrieve(test_query)
        
        if "results" in result:
            print(f"Query: '{test_query}'")
            print("Results distribution:")
            
            total_results = 0
            for category, items in result["results"].items():
                print(f"  {category}: {len(items)} items")
                total_results += len(items)
                
                # Show score distribution
                if items:
                    scores = [item[1] for item in items]
                    avg_score = sum(scores) / len(scores)
                    print(f"    Score range: {min(scores):.3f} - {max(scores):.3f} (avg: {avg_score:.3f})")
            
            print(f"Total results: {total_results}")
            
            # Check if results are too diluted
            if total_results > 15:
                print("⚠️  ISSUE: Results are diluted across too many categories")

def identify_core_problems():
    """Identify the core architectural problems"""
    print("\n=== CORE PROBLEMS IDENTIFIED ===\n")
    
    problems = [
        "1. QUERY MISMATCH: Natural language queries don't map well to rigid entity categories",
        "2. RESULT DILUTION: 25+ results across 4+ categories reduces focus and relevance", 
        "3. BROKEN CONNECTIONS: Many connections point to non-existent objects",
        "4. OVER-CATEGORIZATION: Splitting related info across entities loses context",
        "5. COMPLEXITY OVERHEAD: Multi-stage pipeline adds latency without improving relevance"
    ]
    
    for problem in problems:
        print(f"❌ {problem}")

if __name__ == "__main__":
    analyze_query_rewriting()
    analyze_connection_problems() 
    analyze_search_distribution()
    identify_core_problems()