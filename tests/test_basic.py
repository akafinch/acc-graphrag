"""
Basic tests for the GraphRAG system.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.default import CHUNK_SIZE, CHUNK_OVERLAP
from database.neo4j_client import Neo4jClient
from ingestion.document_processor import DocumentProcessor
from rag.graph_retriever import GraphRetriever
from rag.ollama_client import OllamaClient

class TestNeo4jClient(unittest.TestCase):
    """Test the Neo4j client."""
    
    def setUp(self):
        """Set up the test."""
        self.mock_driver = MagicMock()
        self.mock_session = MagicMock()
        self.mock_driver.session.return_value = self.mock_session
        
        # Create a patcher for GraphDatabase.driver
        self.driver_patcher = patch('neo4j.GraphDatabase.driver')
        self.mock_graph_db_driver = self.driver_patcher.start()
        self.mock_graph_db_driver.return_value = self.mock_driver
        
        # Initialize the client
        self.client = Neo4jClient()
    
    def tearDown(self):
        """Clean up after the test."""
        self.driver_patcher.stop()
    
    def test_connect(self):
        """Test connecting to Neo4j."""
        self.client.connect()
        self.mock_graph_db_driver.assert_called_once()
        self.assertEqual(self.client.driver, self.mock_driver)
    
    def test_close(self):
        """Test closing the connection."""
        self.client.driver = self.mock_driver
        self.client.close()
        self.mock_driver.close.assert_called_once()
    
    def test_run_query(self):
        """Test running a Cypher query."""
        # Prepare mock data
        mock_record = MagicMock()
        mock_record.data.return_value = {"name": "Test"}
        self.mock_session.run.return_value = [mock_record]
        
        # Connect and run query
        self.client.driver = self.mock_driver
        result = self.client.run_query("MATCH (n) RETURN n.name as name", {"param": "value"})
        
        # Verify
        self.mock_session.run.assert_called_once_with("MATCH (n) RETURN n.name as name", {"param": "value"})
        self.assertEqual(result, [{"name": "Test"}])

class TestOllamaClient(unittest.TestCase):
    """Test the Ollama client."""
    
    def setUp(self):
        """Set up the test."""
        self.mock_client = MagicMock()
        
        # Create a patcher for httpx.Client
        self.client_patcher = patch('httpx.Client')
        self.mock_httpx_client = self.client_patcher.start()
        self.mock_httpx_client.return_value = self.mock_client
        
        # Initialize the client
        self.ollama_client = OllamaClient()
    
    def tearDown(self):
        """Clean up after the test."""
        self.client_patcher.stop()
    
    def test_check_status(self):
        """Test checking Ollama status."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        self.mock_client.get.return_value = mock_response
        
        # Call the method
        result = self.ollama_client.check_status()
        
        # Verify
        self.mock_client.get.assert_called_once()
        self.assertTrue(result)
    
    def test_check_status_error(self):
        """Test checking Ollama status when there's an error."""
        # Set up the mock response to raise an exception
        self.mock_client.get.side_effect = Exception("Connection failed")
        
        # Call the method
        result = self.ollama_client.check_status()
        
        # Verify
        self.mock_client.get.assert_called_once()
        self.assertFalse(result)
    
    def test_generate_embedding(self):
        """Test generating an embedding."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        self.mock_client.post.return_value = mock_response
        
        # Call the method
        result = self.ollama_client.generate_embedding("test text")
        
        # Verify
        self.mock_client.post.assert_called_once()
        self.assertEqual(result, [0.1, 0.2, 0.3])

class TestDocumentProcessor(unittest.TestCase):
    """Test the document processor."""
    
    def setUp(self):
        """Set up the test."""
        # Mock the OllamaClient
        self.mock_ollama = MagicMock()
        self.mock_ollama.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Initialize with the mock
        self.processor = DocumentProcessor(ollama_client=self.mock_ollama)
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.processor.chunk_size, CHUNK_SIZE)
        self.assertEqual(self.processor.chunk_overlap, CHUNK_OVERLAP)
        self.assertEqual(self.processor.ollama_client, self.mock_ollama)
    
    @patch('langchain_community.document_loaders.DirectoryLoader')
    def test_load_documents(self, mock_loader_class):
        """Test loading documents."""
        # Set up the mock
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            MagicMock(page_content="Test content", metadata={"source": "test.pdf"})
        ]
        mock_loader_class.return_value = mock_loader
        
        # Call the method
        with patch('pathlib.Path.exists', return_value=True):
            docs = self.processor.load_documents(partner_name="test_partner")
        
        # Verify
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "Test content")
        self.assertEqual(docs[0].metadata["partner"], "test_partner")

class TestGraphRetriever(unittest.TestCase):
    """Test the graph retriever."""
    
    def setUp(self):
        """Set up the test."""
        # Mock the dependencies
        self.mock_neo4j = MagicMock()
        self.mock_ollama = MagicMock()
        self.mock_ollama.generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Initialize with the mocks
        self.retriever = GraphRetriever(
            neo4j_client=self.mock_neo4j,
            ollama_client=self.mock_ollama
        )
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.retriever.neo4j_client, self.mock_neo4j)
        self.assertEqual(self.retriever.ollama_client, self.mock_ollama)
    
    def test_retrieve(self):
        """Test retrieving information."""
        # Set up the mocks
        self.mock_neo4j.run_query.return_value = [
            {"s": {"id": "1", "content": "Test content"}, "score": 0.9}
        ]
        
        # Patch the internal methods
        with patch.object(self.retriever, '_vector_search_sections') as mock_vector_search:
            with patch.object(self.retriever, '_search_entities') as mock_entity_search:
                with patch.object(self.retriever, '_expand_results_with_graph') as mock_expand:
                    # Set return values
                    mock_vector_search.return_value = [{"id": "1", "type": "Section", "score": 0.9}]
                    mock_entity_search.return_value = [{"id": "2", "type": "Concept", "score": 0.8}]
                    mock_expand.return_value = [
                        {"id": "1", "type": "Section", "score": 0.9, "graph_context": {}},
                        {"id": "2", "type": "Concept", "score": 0.8, "graph_context": {}}
                    ]
                    
                    # Call the method
                    results = self.retriever.retrieve("test query")
                    
                    # Verify
                    self.assertEqual(len(results), 2)
                    self.assertEqual(results[0]["id"], "1")
                    self.assertEqual(results[1]["id"], "2")
                    
                    # Check that the embedding was generated
                    self.mock_ollama.generate_embedding.assert_called_once_with("test query")

if __name__ == '__main__':
    unittest.main() 