"""
Unit tests for Ragas retriever adapters.

Tests the adapter logic with mock retriever objects to ensure 
proper Ragas interface implementation.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation.adapters import MentisRetrieverAdapter, SimpleRagAdapter, SummaryRagAdapter, get_retriever_adapter, get_all_adapters


class MockRetriever:
    """Mock retriever for testing"""
    
    def __init__(self, name: str, mock_results: list = None):
        self.name = name
        self.mock_results = mock_results or ["result1", "result2", "result3"]
        self.enter_called = False
        self.exit_called = False
    
    def retrieve(self, query: str, limit: int = 5) -> list:
        """Mock retrieve method"""
        return self.mock_results[:limit]
    
    def __enter__(self):
        self.enter_called = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_called = True


class TestMentisRetrieverAdapter(unittest.TestCase):
    """Test cases for MentisRetrieverAdapter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_results = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        
    @patch('evaluation.adapters.SimpleRag')
    def test_simple_rag_adapter_creation(self, mock_simple_rag):
        """Test SimpleRag adapter creation"""
        mock_retriever = MockRetriever("simple_rag", self.mock_results)
        mock_simple_rag.return_value = mock_retriever
        
        adapter = MentisRetrieverAdapter('simple_rag')
        
        self.assertEqual(adapter.retriever_name, 'simple_rag')
        mock_simple_rag.assert_called_once()
    
    @patch('evaluation.adapters.SummaryRag')  
    def test_summary_rag_adapter_creation(self, mock_summary_rag):
        """Test SummaryRag adapter creation"""
        mock_retriever = MockRetriever("summary_rag", self.mock_results)
        mock_summary_rag.return_value = mock_retriever
        
        adapter = MentisRetrieverAdapter('summary_rag')
        
        self.assertEqual(adapter.retriever_name, 'summary_rag')
        mock_summary_rag.assert_called_once()
    
    def test_invalid_retriever_name(self):
        """Test error handling for invalid retriever names"""
        with self.assertRaises(ValueError) as context:
            MentisRetrieverAdapter('invalid_retriever')
        
        self.assertIn("Unknown retriever", str(context.exception))
    
    @patch('evaluation.adapters.SimpleRag')
    def test_retrieve_method(self, mock_simple_rag):
        """Test the retrieve method delegates correctly"""
        mock_retriever = MockRetriever("simple_rag", self.mock_results)
        mock_simple_rag.return_value = mock_retriever
        
        adapter = MentisRetrieverAdapter('simple_rag')
        
        # Test retrieve with default top_k
        results = adapter.retrieve("test query")
        self.assertEqual(results, self.mock_results[:5])
        
        # Test retrieve with custom top_k
        results = adapter.retrieve("test query", top_k=3)
        self.assertEqual(results, self.mock_results[:3])
        
        # Test retrieve with top_k larger than available results
        results = adapter.retrieve("test query", top_k=10)
        self.assertEqual(results, self.mock_results)
    
    @patch('evaluation.adapters.SimpleRag')
    def test_retrieve_error_handling(self, mock_simple_rag):
        """Test error handling in retrieve method"""
        mock_retriever = Mock()
        mock_retriever.retrieve.side_effect = Exception("Retrieval failed")
        mock_simple_rag.return_value = mock_retriever
        
        adapter = MentisRetrieverAdapter('simple_rag')
        
        results = adapter.retrieve("test query")
        self.assertEqual(results, [])
    
    @patch('evaluation.adapters.SimpleRag')
    def test_context_manager(self, mock_simple_rag):
        """Test context manager functionality"""
        mock_retriever = MockRetriever("simple_rag", self.mock_results)
        mock_simple_rag.return_value = mock_retriever
        
        adapter = MentisRetrieverAdapter('simple_rag')
        
        with adapter:
            self.assertTrue(mock_retriever.enter_called)
        
        self.assertTrue(mock_retriever.exit_called)
    
    @patch('evaluation.adapters.SimpleRag')
    def test_context_manager_no_support(self, mock_simple_rag):
        """Test context manager with retriever that doesn't support it"""
        mock_retriever = Mock()
        del mock_retriever.__enter__
        del mock_retriever.__exit__
        mock_simple_rag.return_value = mock_retriever
        
        adapter = MentisRetrieverAdapter('simple_rag')
        
        # Should not raise error even if underlying retriever doesn't support context manager
        try:
            with adapter:
                pass
        except AttributeError:
            self.fail("Context manager should handle missing __enter__/__exit__ methods")


class TestConvenienceAdapters(unittest.TestCase):
    """Test convenience adapter classes"""
    
    @patch('evaluation.adapters.SimpleRag')
    def test_simple_rag_adapter(self, mock_simple_rag):
        """Test SimpleRagAdapter convenience class"""
        mock_retriever = MockRetriever("simple_rag")
        mock_simple_rag.return_value = mock_retriever
        
        adapter = SimpleRagAdapter()
        
        self.assertEqual(adapter.retriever_name, 'simple_rag')
    
    @patch('evaluation.adapters.SummaryRag')
    def test_summary_rag_adapter(self, mock_summary_rag):
        """Test SummaryRagAdapter convenience class"""
        mock_retriever = MockRetriever("summary_rag")
        mock_summary_rag.return_value = mock_retriever
        
        adapter = SummaryRagAdapter()
        
        self.assertEqual(adapter.retriever_name, 'summary_rag')


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions"""
    
    @patch('evaluation.adapters.SimpleRag')
    def test_get_retriever_adapter(self, mock_simple_rag):
        """Test get_retriever_adapter factory function"""
        mock_retriever = MockRetriever("simple_rag")
        mock_simple_rag.return_value = mock_retriever
        
        adapter = get_retriever_adapter('simple_rag')
        
        self.assertIsInstance(adapter, MentisRetrieverAdapter)
        self.assertEqual(adapter.retriever_name, 'simple_rag')
    
    @patch('evaluation.adapters.SimpleRag')
    @patch('evaluation.adapters.SummaryRag')
    def test_get_all_adapters(self, mock_summary_rag, mock_simple_rag):
        """Test get_all_adapters factory function"""
        mock_simple_rag.return_value = MockRetriever("simple_rag")
        mock_summary_rag.return_value = MockRetriever("summary_rag")
        
        adapters = get_all_adapters()
        
        self.assertEqual(len(adapters), 2)
        self.assertIn('simple_rag', adapters)
        self.assertIn('summary_rag', adapters)
        self.assertIsInstance(adapters['simple_rag'], SimpleRagAdapter)
        self.assertIsInstance(adapters['summary_rag'], SummaryRagAdapter)


class TestRagasInterface(unittest.TestCase):
    """Test Ragas interface compliance"""
    
    @patch('evaluation.adapters.SimpleRag')
    def test_ragas_retrieve_signature(self, mock_simple_rag):
        """Test that retrieve method matches Ragas expectations"""
        mock_retriever = MockRetriever("simple_rag", ["doc1", "doc2", "doc3"])
        mock_simple_rag.return_value = mock_retriever
        
        adapter = MentisRetrieverAdapter('simple_rag')
        
        # Test that method exists with correct signature
        self.assertTrue(hasattr(adapter, 'retrieve'))
        self.assertTrue(callable(adapter.retrieve))
        
        # Test return type
        results = adapter.retrieve("test query", top_k=2)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], str)
    
    @patch('evaluation.adapters.SimpleRag')
    def test_empty_results(self, mock_simple_rag):
        """Test handling of empty retrieval results"""
        mock_retriever = MockRetriever("simple_rag", [])
        mock_simple_rag.return_value = mock_retriever
        
        adapter = MentisRetrieverAdapter('simple_rag')
        
        results = adapter.retrieve("test query")
        self.assertEqual(results, [])
        self.assertIsInstance(results, list)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)