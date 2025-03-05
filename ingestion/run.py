#!/usr/bin/env python
"""
Script to run the document ingestion process.
"""

import argparse
import logging
import os
import sys
import json
import re
from pathlib import Path

from ingestion.document_processor import DocumentProcessor
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
    parser.add_argument(
        "--skip-entity-extraction", 
        action="store_true", 
        help="Skip the entity extraction step (faster processing)"
    )
    parser.add_argument(
        "--entity-limit", 
        type=int,
        default=100, 
        help="Maximum number of chunks to process for entity extraction"
    )
    parser.add_argument(
        "--entity-timeout", 
        type=int,
        default=60, 
        help="Timeout in seconds per chunk for entity extraction"
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

def load_to_graph(partner_name: str, processed_dir: str) -> None:
    """Load processed data to graph database.
    
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
        
        # First, clear any existing data for this partner to avoid duplication
        clear_query = """
        MATCH (p:Partner {id: $partner_id})-[:HAS_DOCUMENT]->(d:Document)
        OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
        DETACH DELETE s, d
        """
        try:
            neo4j_client.run_query(clear_query, {"partner_id": f"partner-{partner_name}"})
            
            # Also clear any orphaned sections that might have conflicting IDs
            clear_orphans_query = """
            MATCH (s:Section)
            WHERE NOT EXISTS ((s)<-[:HAS_SECTION]-())
            DETACH DELETE s
            """
            neo4j_client.run_query(clear_orphans_query)
            
            logger.info(f"Cleared existing document data for partner {partner_name}")
        except Exception as e:
            logger.warning(f"Could not clear existing data: {str(e)}")
        
        # Load the JSON data
        with open(latest_doc_file, 'r') as f:
            data = json.load(f)
            
        # Create partner node if it doesn't exist
        partner_props = {
            "id": f"partner-{partner_name}",
            "name": partner_name,
            "description": f"Documentation for {partner_name}"
        }
        partner_node = neo4j_client.create_node("Partner", partner_props)
        logger.info(f"Created/updated partner node: {partner_node}")
        
        # Initialize counters for documents and chunks
        chunk_count = 0
        doc_count = 0
        
        # Handle the specific structure where there's just a 'chunks' key with a list of chunks
        if isinstance(data, dict) and 'chunks' in data and isinstance(data['chunks'], list):
            chunks = data['chunks']
            
            # Group chunks by source document using metadata
            chunks_by_source = {}
            logger.info(f"Processing {len(chunks)} chunks from the data file")
            
            for idx, chunk in enumerate(chunks):
                metadata = chunk.get('metadata', {})
                # Try to extract a useful source identifier from metadata
                source = None
                
                # Check various possible metadata fields for document identification
                if isinstance(metadata, dict):
                    # Try to get the source from various possible metadata fields
                    source = metadata.get('source') or metadata.get('url') or metadata.get('file') or metadata.get('document')
                    
                    # If there's a title or filename, use that as part of the source
                    title = metadata.get('title') or metadata.get('filename')
                    if title and source:
                        source = f"{source}:{title}"
                    elif title:
                        source = title
                
                # If no source found in metadata, create a grouping by index
                if not source:
                    # Group every 5 chunks together as a "document" (reduced from 10 to create more docs)
                    source = f"document-{idx // 5 + 1}"
                
                # Add chunk to the appropriate source group
                if source not in chunks_by_source:
                    chunks_by_source[source] = []
                    logger.info(f"Created new document group: {source}")
                chunks_by_source[source].append((idx, chunk))
                
                # Periodically log progress for large chunk lists
                if idx % 200 == 0 and idx > 0:
                    logger.info(f"Processed {idx}/{len(chunks)} chunks so far, created {len(chunks_by_source)} document groups")
            
            logger.info(f"Finished grouping chunks into {len(chunks_by_source)} documents")
            
            # Now create documents and sections for each source group
            doc_count = 0
            for source, source_chunks in chunks_by_source.items():
                # Create a document for each source
                doc_id = f"doc-{slugify(source)}"
                title = source
                
                # If the title is too long, truncate it
                if len(title) > 100:
                    title = title[:97] + "..."
                
                try:
                    # Create direct Cypher query to create the document node
                    create_doc_query = """
                    MERGE (d:Document {id: $id})
                    SET d.title = $title,
                        d.source = $source,
                        d.partner = $partner
                    RETURN d
                    """
                    
                    doc_params = {
                        "id": doc_id,
                        "title": title,
                        "source": source,
                        "partner": partner_name
                    }
                    
                    doc_result = neo4j_client.run_query(create_doc_query, doc_params)
                    doc_count += 1
                    
                    # Create relationship from partner to document
                    partner_rel_query = """
                    MATCH (p:Partner {id: $partner_id}), (d:Document {id: $doc_id})
                    MERGE (p)-[:HAS_DOCUMENT]->(d)
                    """
                    
                    partner_rel_params = {
                        "partner_id": f"partner-{partner_name}",
                        "doc_id": doc_id
                    }
                    
                    neo4j_client.run_query(partner_rel_query, partner_rel_params)
                    
                    # Log how many chunks we're about to process
                    chunk_count_for_doc = len(source_chunks)
                    logger.info(f"Processing {chunk_count_for_doc} sections for document {doc_id}")
                    
                    # Create sections and relationships in a single query per section
                    # This is more reliable than batching
                    sections_created = 0
                    
                    for i, (chunk_idx, chunk) in enumerate(source_chunks):
                        content = chunk.get('content', '')
                        if not content:
                            continue
                        
                        chunk_id = f"{doc_id}-chunk-{chunk_idx}"
                        
                        # Create section and relationship in a single query
                        create_section_query = """
                        MATCH (d:Document {id: $doc_id})
                        MERGE (s:Section {id: $section_id})
                        SET s.content = $content,
                            s.index = $index
                        MERGE (d)-[:HAS_SECTION]->(s)
                        RETURN s
                        """
                        
                        # Don't include embeddings in the query parameters directly
                        # they're too large and can cause issues
                        section_params = {
                            "doc_id": doc_id,
                            "section_id": chunk_id,
                            "content": content,
                            "index": chunk_idx
                        }
                        
                        # Add metadata as separate properties if available
                        metadata = chunk.get('metadata', {})
                        if isinstance(metadata, dict):
                            # Check if there's an 'id' in metadata and log it but don't use it
                            # This prevents ID collisions with our generated IDs
                            if 'id' in metadata:
                                logger.warning(f"Found 'id' in chunk metadata: {metadata['id']} - using our generated ID instead: {chunk_id}")
                                # Add the original ID as a different property
                                section_params['original_id'] = metadata['id']
                                create_section_query = create_section_query.replace(
                                    "SET s.content = $content,",
                                    "SET s.content = $content, s.original_id = $original_id,"
                                )
                            
                            # Add the rest of the metadata as properties
                            for key, value in metadata.items():
                                if key not in ["content", "embedding", "id"] and isinstance(value, (str, int, float, bool)):
                                    section_params[key] = value
                                    # Update the query to set this property
                                    create_section_query = create_section_query.replace(
                                        "SET s.content = $content,",
                                        f"SET s.content = $content, s.{key} = ${key},"
                                    )
                        
                        try:
                            result = neo4j_client.run_query(create_section_query, section_params)
                            sections_created += 1
                            chunk_count += 1
                            
                            # Log progress periodically
                            if sections_created % 10 == 0 or sections_created == chunk_count_for_doc:
                                logger.info(f"Created {sections_created}/{chunk_count_for_doc} sections for document {doc_id}")
                        
                        except Exception as e:
                            logger.error(f"Error creating section {chunk_id}: {str(e)}")
                    
                    logger.info(f"Completed processing document {doc_id} with {sections_created} sections")
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {str(e)}")
        else:
            logger.warning(f"Unexpected data format. Expected a dictionary with 'chunks' key.")
        
        logger.info(f"Loaded {doc_count} documents with {chunk_count} chunks into the graph")
        
        # Final step: ensure all documents for this partner are connected to the partner node
        connect_docs_query = """
        MATCH (d:Document) 
        WHERE d.partner = $partner_name AND NOT EXISTS((d)<-[:HAS_DOCUMENT]-(:Partner {id: $partner_id}))
        WITH collect(d.id) as missing_connections, count(d) as missing_count
        RETURN missing_connections, missing_count
        """
        
        result = neo4j_client.run_query(connect_docs_query, {
            "partner_name": partner_name,
            "partner_id": f"partner-{partner_name}"
        })
        
        if result and result[0]["missing_count"] > 0:
            missing_docs = result[0]["missing_connections"]
            logger.warning(f"Found {result[0]['missing_count']} documents not connected to partner. Fixing...")
            
            # Create connections for all missing documents
            fix_connections_query = """
            MATCH (p:Partner {id: $partner_id})
            MATCH (d:Document)
            WHERE d.id IN $doc_ids
            MERGE (p)-[:HAS_DOCUMENT]->(d)
            RETURN count(*) as fixed_count
            """
            
            fix_result = neo4j_client.run_query(fix_connections_query, {
                "partner_id": f"partner-{partner_name}",
                "doc_ids": missing_docs
            })
            
            if fix_result:
                logger.info(f"Fixed {fix_result[0]['fixed_count']} document-to-partner connections")
        else:
            logger.info("All documents are properly connected to the partner node")
    
    # Load the most recent entities file
    entity_files = list(processed_path.glob("*_entities_*.json"))
    entity_files.sort(reverse=True)
    
    if entity_files:
        latest_entity_file = entity_files[0]
        logger.info(f"Loading entities from {latest_entity_file}")
        
        # Implement loading entities to graph
        with open(latest_entity_file, 'r') as f:
            entity_data = json.load(f)
            
        # Track entity counts by type
        entity_counts = {}
        relationship_count = 0
        
        # Check if entity_data is a list or dictionary and process accordingly
        if isinstance(entity_data, list):
            # Handle list format - each item is a document with entities
            for doc_entry in entity_data:
                doc_id = doc_entry.get("doc_id", "unknown-doc")
                entities = doc_entry.get("entities", {})
                
                if not entities:
                    continue
                    
                # Process each type of entity
                for entity_type, entity_list in entities.items():
                    if not entity_list:
                        continue
                        
                    # Initialize count for this entity type
                    if entity_type not in entity_counts:
                        entity_counts[entity_type] = 0
                        
                    # Process each entity of this type
                    for entity in entity_list:
                        # Create and process entity as before
                        entity_name = entity.get("name", "Unknown")
                        entity_id = f"{entity_type.lower()}-{slugify(entity_name)}"
                        
                        entity_props = {
                            "id": entity_id,
                            "name": entity_name,
                            "description": entity.get("description", ""),
                            "source_doc": doc_id
                        }
                        
                        # Add any additional properties
                        for key, value in entity.items():
                            if key not in ["name", "description", "relationships"] and value:
                                entity_props[key] = value
                        
                        # Create the entity node
                        entity_node = neo4j_client.create_node(entity_type, entity_props)
                        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                        
                        try:
                            # Create relationship from document to entity
                            neo4j_client.create_relationship(
                                doc_id,
                                entity_props["id"],
                                "MENTIONS"
                            )
                            relationship_count += 1
                        except Exception as e:
                            logger.warning(f"Could not create MENTIONS relationship: {str(e)}")
                        
                        # Create relationship from partner to entity
                        neo4j_client.create_relationship(
                            f"partner-{partner_name}",
                            entity_props["id"],
                            "HAS_ENTITY"
                        )
                        relationship_count += 1
                        
                        # If entity has relationships to other entities, create those too
                        if "relationships" in entity and entity["relationships"]:
                            for rel in entity["relationships"]:
                                if "target" in rel and "type" in rel:
                                    target_id = f"{rel.get('target_type', 'concept').lower()}-{slugify(rel['target'])}"
                                    rel_type = rel["type"].upper().replace(" ", "_")
                                    
                                    # Create the relationship
                                    try:
                                        neo4j_client.create_relationship(
                                            entity_props["id"],
                                            target_id,
                                            rel_type,
                                            properties=rel.get("properties", {})
                                        )
                                        relationship_count += 1
                                    except Exception as e:
                                        logger.warning(f"Could not create relationship: {str(e)}")
        else:
            # Handle dictionary format - keys are doc_ids, values are entity dictionaries
            for doc_id, entities in entity_data.items():
                # Skip if no entities were extracted
                if not entities:
                    continue
                    
                for entity_type, entity_list in entities.items():
                    if not entity_list:
                        continue
                        
                    # Initialize count for this entity type
                    if entity_type not in entity_counts:
                        entity_counts[entity_type] = 0
                        
                    # Process each entity of this type
                    for entity in entity_list:
                        # Create entity node with properties
                        entity_name = entity.get("name", "Unknown")
                        entity_id = f"{entity_type.lower()}-{slugify(entity_name)}"
                        
                        entity_props = {
                            "id": entity_id,
                            "name": entity_name,
                            "description": entity.get("description", ""),
                            "source_doc": doc_id
                        }
                        
                        # Add any additional properties
                        for key, value in entity.items():
                            if key not in ["name", "description", "relationships"] and value:
                                entity_props[key] = value
                        
                        # Create the entity node
                        entity_node = neo4j_client.create_node(entity_type, entity_props)
                        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                        
                        try:
                            # Create relationship from document to entity
                            neo4j_client.create_relationship(
                                doc_id,
                                entity_props["id"],
                                "MENTIONS"
                            )
                            relationship_count += 1
                        except Exception as e:
                            logger.warning(f"Could not create MENTIONS relationship: {str(e)}")
                        
                        # Create relationship from partner to entity
                        neo4j_client.create_relationship(
                            f"partner-{partner_name}",
                            entity_props["id"],
                            "HAS_ENTITY"
                        )
                        relationship_count += 1
                        
                        # If entity has relationships to other entities, create those too
                        if "relationships" in entity and entity["relationships"]:
                            for rel in entity["relationships"]:
                                if "target" in rel and "type" in rel:
                                    target_id = f"{rel.get('target_type', 'concept').lower()}-{slugify(rel['target'])}"
                                    rel_type = rel["type"].upper().replace(" ", "_")
                                    
                                    # Create the relationship
                                    try:
                                        neo4j_client.create_relationship(
                                            entity_props["id"],
                                            target_id,
                                            rel_type,
                                            properties=rel.get("properties", {})
                                        )
                                        relationship_count += 1
                                    except Exception as e:
                                        logger.warning(f"Could not create relationship: {str(e)}")
        
        # Log the entity counts
        for entity_type, count in entity_counts.items():
            logger.info(f"Loaded {count} {entity_type} entities")
        logger.info(f"Created {relationship_count} relationships")
    
    logger.info(f"Completed loading data for partner {partner_name} to graph")

# Helper function to create URL-friendly slug from entity names
def slugify(text):
    """Convert text to a URL-friendly slug."""
    # Remove special characters and replace spaces with hyphens
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[\s_-]+', '-', text)
    return text

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
            chunks, entities = document_processor.process_partner_documents(
                partner_name,
                skip_entity_extraction=args.skip_entity_extraction,
                entity_sample_limit=args.entity_limit,
                entity_timeout=args.entity_timeout
            )
        else:
            logger.info(f"Processing documents for partner {partner_name}")
            chunks, entities = document_processor.process_partner_documents(
                partner_name,
                skip_entity_extraction=args.skip_entity_extraction,
                entity_sample_limit=args.entity_limit,
                entity_timeout=args.entity_timeout
            )
        
        logger.info(f"Processed {len(chunks)} document chunks and extracted {len(entities)} entities")
        
        # Load to graph if requested
        if args.load_to_graph:
            load_to_graph(partner_name, document_processor.processed_data_dir)
    else:
        logger.error("No partner specified. Use --partner NAME or --list-partners")
        sys.exit(1)

if __name__ == "__main__":
    main() 