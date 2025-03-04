#!/usr/bin/env python
"""
Script to run the document ingestion process.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from document_processor import DocumentProcessor
from database.neo4j_client import Neo4jClient
from rag.ollama_client import OllamaClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the document ingestion process")
    parser.add_argument(
        "--partner", 
        type=str, 
        help="Name of the partner to process documents for"
    )
    parser.add_argument(
        "--source-dir", 
        type=str, 
        help="Source directory for documents (overrides default)"
    )
    parser.add_argument(
        "--load-to-graph", 
        action="store_true", 
        help="Load processed data to the graph database"
    )
    parser.add_argument(
        "--list-partners", 
        action="store_true", 
        help="List partners with documents in the raw data directory"
    )
    
    return parser.parse_args()

def list_partners(raw_data_dir):
    """List partners with documents in the raw data directory."""
    raw_data_path = Path(raw_data_dir)
    
    if not raw_data_path.exists():
        logger.error(f"Raw data directory does not exist: {raw_data_path}")
        return []
    
    partners = [item.name for item in raw_data_path.iterdir() if item.is_dir()]
    return partners

def load_to_graph(partner_name, processed_dir):
    """
    Load processed data to the graph database.
    
    Args:
        partner_name: Name of the partner
        processed_dir: Directory with processed data
    """
    logger.info(f"Loading data for partner {partner_name} to graph")
    
    # Connect to Neo4j
    neo4j_client = Neo4jClient()
    neo4j_client.connect()
    
    processed_path = Path(processed_dir) / partner_name
    
    if not processed_path.exists():
        logger.error(f"No processed data found for partner: {partner_name}")
        return
    
    # Setup database schema
    neo4j_client.setup_database()
    
    # Load the most recent processed documents file
    document_files = list(processed_path.glob("*_processed_*.json"))
    document_files.sort(reverse=True)
    
    if document_files:
        latest_doc_file = document_files[0]
        logger.info(f"Loading document chunks from {latest_doc_file}")
        
        # TODO: Implement loading document chunks to graph
        
    # Load the most recent entities file
    entity_files = list(processed_path.glob("*_entities_*.json"))
    entity_files.sort(reverse=True)
    
    if entity_files:
        latest_entity_file = entity_files[0]
        logger.info(f"Loading entities from {latest_entity_file}")
        
        # TODO: Implement loading entities to graph
    
    logger.info(f"Completed loading data for partner {partner_name} to graph")

def main():
    """Run the document ingestion process."""
    args = parse_args()
    
    # Initialize components
    ollama_client = OllamaClient()
    
    # Check Ollama status
    if not ollama_client.check_status():
        logger.error("Ollama is not running. Please start Ollama first.")
        sys.exit(1)
    
    # List available models
    available_models = ollama_client.list_models()
    required_models = [ollama_client.llm_model, ollama_client.embedding_model]
    
    missing_models = [model for model in required_models if model not in available_models]
    if missing_models:
        logger.warning(f"Required models not available: {', '.join(missing_models)}")
        logger.info("Please run:")
        for model in missing_models:
            logger.info(f"  ollama pull {model}")
        sys.exit(1)
    
    # Initialize document processor
    document_processor = DocumentProcessor(ollama_client=ollama_client)
    
    # List partners
    if args.list_partners:
        partners = list_partners(document_processor.raw_data_dir)
        if partners:
            logger.info("Available partners:")
            for partner in partners:
                logger.info(f"  - {partner}")
        else:
            logger.info("No partners found in raw data directory")
        return
    
    # Process partner documents
    if args.partner:
        partner_name = args.partner
        source_dir = args.source_dir
        
        if source_dir:
            logger.info(f"Processing documents for partner {partner_name} from {source_dir}")
            chunks, entities = document_processor.process_partner_documents(partner_name)
        else:
            logger.info(f"Processing documents for partner {partner_name}")
            chunks, entities = document_processor.process_partner_documents(partner_name)
        
        logger.info(f"Processed {len(chunks)} document chunks and extracted {len(entities)} entities")
        
        # Load to graph if requested
        if args.load_to_graph:
            load_to_graph(partner_name, document_processor.processed_data_dir)
    else:
        logger.error("No partner specified. Use --partner NAME or --list-partners")
        sys.exit(1)

if __name__ == "__main__":
    main() 