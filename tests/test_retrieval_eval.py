"""
Ragas-based retrieval evaluation for Mentis project.

Evaluates SimpleRag and SummaryRag retrievers using context recall, 
context precision, and per-chunk relevance metrics.
"""

import json
import os
import sys
from typing import List, Dict, Any

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation.adapters import get_all_adapters, MentisRetrieverAdapter

try:
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import ContextRelevance, LLMContextPrecisionWithoutReference
    from datasets import Dataset as HFDataset
except ImportError as e:
    print(f"Error importing ragas: {e}")
    print("Please install ragas with: pip install ragas")
    sys.exit(1)


class RetrievalEvaluator:
    """Main evaluation class for retrieval systems"""
    
    def __init__(self, queries_file: str = "evaluation/queries.json"):
        """
        Initialize evaluator with query file.
        
        Args:
            queries_file: Path to JSON file with evaluation queries
        """
        self.queries_file = queries_file
        self.metrics = [ContextRelevance()]
        
    def load_queries(self) -> List[str]:
        """Load evaluation queries from JSON file"""
        try:
            with open(self.queries_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'queries' in data:
                return data['queries']
            else:
                raise ValueError("Invalid queries file format")
                
        except FileNotFoundError:
            print(f"Queries file not found: {self.queries_file}")
            # Return sample queries for testing
            return [
                "What activities did the user do with friends?",
                "What work-related challenges did the user face?",
                "What emotions did the user express about family relationships?"
            ]
    
    
    def create_ragas_dataset(self, retriever: MentisRetrieverAdapter, 
                           queries: List[str], top_k: int = 5) -> EvaluationDataset:
        """
        Create a Ragas Dataset for evaluation without gold standards.
        
        Args:
            retriever: Retriever adapter to evaluate
            queries: List of evaluation queries
            top_k: Number of documents to retrieve per query
            
        Returns:
            Ragas Dataset ready for evaluation
        """
        # Prepare data for Ragas (only need query and retrieved contexts)
        eval_data = {
            'user_input': [],
            'retrieved_contexts': []
        }
        
        with retriever:
            for query in queries:
                # Retrieve documents
                retrieved_docs = retriever.retrieve(query, top_k=top_k)
                
                if retrieved_docs:  # Only add if we got results
                    eval_data['user_input'].append(query)
                    eval_data['retrieved_contexts'].append(retrieved_docs)
        
        # Convert to HuggingFace Dataset format that Ragas expects
        hf_dataset = HFDataset.from_dict(eval_data)
        return EvaluationDataset.from_hf_dataset(hf_dataset)
    
    def evaluate_retriever(self, retriever_name: str, retriever: MentisRetrieverAdapter,
                          queries: List[str]) -> Dict[str, Any]:
        """
        Evaluate a single retriever using Ragas metrics without gold standards.
        
        Args:
            retriever_name: Name of the retriever
            retriever: Retriever adapter instance
            queries: List of evaluation queries
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\nEvaluating {retriever_name}...")
        
        try:
            # Create dataset
            dataset = self.create_ragas_dataset(retriever, queries)
            
            if len(dataset) == 0:
                print(f"No valid queries found for {retriever_name}")
                return {"error": "No valid data"}
            
            print(f"Created dataset with {len(dataset)} samples")
            
            # Run evaluation
            results = evaluate(dataset, metrics=self.metrics)
            
            # Extract results (handle different result formats)
            try:
                # Get the metric names from our metrics
                metric_names = [metric.name for metric in self.metrics]
                
                result_dict = {
                    "retriever": retriever_name,
                    "num_queries": len(dataset),
                    "raw_results": str(results)
                }
                
                # Extract results - ragas 0.3.0 stores results directly accessible by metric names
                if hasattr(results, '__getitem__'):
                    # Try to get the context relevance score (which we know exists from raw output)
                    try:
                        # The metric name in ragas 0.3.0 appears to be 'nv_context_relevance'
                        if 'nv_context_relevance' in str(results):
                            import re
                            raw_str = str(results)
                            match = re.search(r"'nv_context_relevance': ([\d.]+)", raw_str)
                            if match:
                                result_dict['nv_context_relevance'] = float(match.group(1))
                        
                        # Also try direct access if possible
                        try:
                            score = results['nv_context_relevance']
                            result_dict['nv_context_relevance'] = float(score)
                        except (KeyError, TypeError):
                            pass
                            
                    except Exception as e:
                        print(f"Could not extract metric scores: {e}")
                else:
                    print(f"Unexpected results format: {type(results)}")
                
            except (KeyError, TypeError, ValueError) as e:
                print(f"Error processing results for {retriever_name}: {e}")
                result_dict = {
                    "retriever": retriever_name,
                    "num_queries": len(dataset),
                    "raw_results": str(results)
                }
                # Add default values for metrics
                for metric in self.metrics:
                    result_dict[metric.name] = 0.0
            
            return result_dict
            
        except Exception as e:
            print(f"Error evaluating {retriever_name}: {e}")
            return {"error": str(e)}
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation on all retrievers without gold standards.
        
        Returns:
            Dictionary with all evaluation results
        """
        print("Starting Mentis retrieval evaluation...")
        
        # Load queries
        queries = self.load_queries()
        print(f"Loaded {len(queries)} queries")
        
        # Get retrievers
        adapters = get_all_adapters()
        
        # Run evaluation for each retriever
        all_results = {}
        
        for name, adapter in adapters.items():
            result = self.evaluate_retriever(name, adapter, queries)
            all_results[name] = result
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "evaluation"):
        """Save evaluation results to JSON files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for retriever_name, result in results.items():
            if "error" not in result:
                output_file = os.path.join(output_dir, f"results_{retriever_name}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Saved {retriever_name} results to {output_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary table of results"""
        print("\n" + "="*80)
        print("RETRIEVAL EVALUATION SUMMARY")
        print("="*80)
        
        print(f"{'Retriever':<15} {'Queries':<8} {'Context Relevance':<15}")
        print("-" * 60)
        
        for name, result in results.items():
            if "error" in result:
                print(f"{name:<15} ERROR: {result['error']}")
            else:
                relevance = result.get('nv_context_relevance', result.get('context_relevance', 0))
                num_queries = result.get('num_queries', 0)
                
                print(f"{name:<15} {num_queries:<8} {relevance:<15.3f}")


def get_metrics() -> Dict[str, Any]:
    """
    Helper function that can be imported by evaluation.py.
    
    Returns:
        Dictionary with latest evaluation metrics
    """
    evaluator = RetrievalEvaluator()
    return evaluator.run_evaluation()


if __name__ == "__main__":
    # Run evaluation
    evaluator = RetrievalEvaluator()
    results = evaluator.run_evaluation()
    
    if results:
        evaluator.save_results(results)
        evaluator.print_summary(results)
    else:
        print("No results to display")