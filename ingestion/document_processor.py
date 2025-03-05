"""
Document processor for ingesting partner documents.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import langchain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (DirectoryLoader,
                                               PyPDFLoader,
                                               TextLoader, UnstructuredPDFLoader)
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.default import (CHUNK_OVERLAP, CHUNK_SIZE, PROCESSED_DATA_DIR,
                           RAW_DATA_DIR)
from rag.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Process documents for ingestion into the GraphRAG system.
    """
    
    def __init__(
        self,
        raw_data_dir: str = RAW_DATA_DIR,
        processed_data_dir: str = PROCESSED_DATA_DIR,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        ollama_client: Optional[OllamaClient] = None
    ):
        """Initialize the document processor."""
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_client = ollama_client or OllamaClient()
        
        # Ensure directories exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_documents(
        self, 
        source_dir: Optional[Union[str, Path]] = None, 
        partner_name: Optional[str] = None,
        file_ext: List[str] = [".pdf", ".txt", ".md", ".html"]
    ) -> List[Document]:
        """
        Load documents from a directory.
        
        Args:
            source_dir: Directory containing documents (default: raw_data_dir)
            partner_name: Subdirectory for a specific partner
            file_ext: List of file extensions to load
        
        Returns:
            List of loaded documents
        """
        if source_dir is None:
            source_dir = self.raw_data_dir
        else:
            source_dir = Path(source_dir)
        
        if partner_name:
            source_dir = source_dir / partner_name
        
        if not source_dir.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return []
        
        documents = []
        
        # Process each supported file type
        for ext in file_ext:
            if ext == ".pdf":
                loader = DirectoryLoader(
                    source_dir, 
                    glob=f"**/*{ext}", 
                    loader_cls=PyPDFLoader
                )
            elif ext == ".txt":
                loader = DirectoryLoader(
                    source_dir, 
                    glob=f"**/*{ext}", 
                    loader_cls=TextLoader
                )
            else:
                # Use unstructured loader for other formats
                loader = DirectoryLoader(
                    source_dir, 
                    glob=f"**/*{ext}"
                )
            
            try:
                docs = loader.load()
                if docs:
                    logger.info(f"Loaded {len(docs)} {ext} documents from {source_dir}")
                    documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {ext} documents: {str(e)}")
        
        for doc in documents:
            # Add metadata
            if partner_name and "partner" not in doc.metadata:
                doc.metadata["partner"] = partner_name
            
            # Ensure each document has a unique ID
            if "id" not in doc.metadata:
                doc.metadata["id"] = str(uuid.uuid4())
        
        return documents
    
    def split_documents(
        self, 
        documents: List[Document]
    ) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
        
        Returns:
            List of document chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for doc in tqdm(documents, desc="Splitting documents"):
            try:
                doc_chunks = splitter.split_documents([doc])
                
                # Add section numbers and preserve metadata
                for i, chunk in enumerate(doc_chunks):
                    # Preserve original metadata
                    for key, value in doc.metadata.items():
                        if key not in chunk.metadata:
                            chunk.metadata[key] = value
                    
                    # Add section metadata
                    chunk.metadata["section_id"] = f"{doc.metadata.get('id', 'doc')}_s{i}"
                    chunk.metadata["section_number"] = i
                    
                chunks.extend(doc_chunks)
            except Exception as e:
                logger.error(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {str(e)}")
        
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(
        self, 
        chunks: List[Document]
    ) -> List[Document]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
        
        Returns:
            List of document chunks with embeddings
        """
        for chunk in tqdm(chunks, desc="Generating embeddings"):
            try:
                embedding = self.ollama_client.generate_embedding(chunk.page_content)
                chunk.metadata["embedding"] = embedding
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {chunk.metadata.get('section_id', 'unknown')}: {str(e)}")
        
        return chunks
    
    def extract_entities(
        self, 
        chunks: List[Document],
        batch_size: int = 5,  # Process chunks in batches
        timeout_per_chunk: int = 60  # 60 second timeout per chunk
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from document chunks using LLM.
        
        Args:
            chunks: List of document chunks
            batch_size: Number of chunks to process at once
            timeout_per_chunk: Maximum seconds to spend per chunk
        
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        system_prompt = """
        You are an AI assistant specialized in extracting structured information from technical documentation.
        Identify the following types of entities in the text:
        1. Concepts - Key technical concepts or ideas
        2. Products - Specific products or offerings mentioned
        3. Services - Cloud services or capabilities
        4. Technologies - Technologies, frameworks, or standards
        5. Use Cases - Specific scenarios or applications
        
        For each entity identified, provide:
        - name: The entity name
        - type: One of [Concept, Product, Service, Technology, UseCase]
        - description: Brief description based on the context
        - relationships: Other entities it relates to in the text
        """
        
        prompt_template = """
        Extract structured entities from the following text:
        
        {text}
        
        Respond ONLY with a JSON object containing the identified entities with this structure:
        {{
          "entities": [
            {{
              "name": "entity name",
              "type": "entity type (Concept, Product, Service, Technology, UseCase)",
              "description": "brief description from context",
              "relationships": [
                {{
                  "target": "related entity name",
                  "relation_type": "relationship type (RELATES_TO, IMPLEMENTS, DEPENDS_ON, PART_OF, ENABLES, ALTERNATIVE_TO)"
                }}
              ]
            }}
          ]
        }}
        
        Remember to ONLY return valid JSON.
        """
        
        # Process only a random sample of chunks if there are too many
        if len(chunks) > 100:
            logging.warning(f"Large number of chunks ({len(chunks)}). Processing a representative sample for entities.")
            # Take a representative sample of chunks (at most 100)
            import random
            random.seed(42)  # For reproducibility
            sampled_chunks = random.sample(chunks, min(100, len(chunks)))
            chunks_to_process = sampled_chunks
        else:
            chunks_to_process = chunks
            
        # Process chunks in batches
        total_chunks = len(chunks_to_process)
        for i in range(0, total_chunks, batch_size):
            batch = chunks_to_process[i:i+batch_size]
            
            # Process batch with progress information
            logging.info(f"Processing entity batch {i//batch_size + 1} of {(total_chunks + batch_size - 1)//batch_size} ({i}/{total_chunks} chunks)")
            
            for chunk_idx, chunk in enumerate(tqdm(batch, desc=f"Extracting entities (batch {i//batch_size + 1})")):
                try:
                    # Skip very short chunks (likely not meaningful content)
                    if len(chunk.page_content.strip()) < 100:
                        logging.info(f"Skipping short chunk ({len(chunk.page_content)} chars)")
                        continue
                    
                    prompt = prompt_template.format(text=chunk.page_content[:1500])  # Limit text size
                    
                    # Set timeout for this operation
                    import threading
                    import time
                    
                    response = None
                    extraction_error = None
                    
                    def extract_with_timeout():
                        nonlocal response, extraction_error
                        try:
                            response = self.ollama_client.generate_text(
                                prompt=prompt,
                                system_prompt=system_prompt,
                                temperature=0.2
                            )
                        except Exception as e:
                            extraction_error = e
                    
                    # Run extraction with timeout
                    thread = threading.Thread(target=extract_with_timeout)
                    thread.start()
                    thread.join(timeout=timeout_per_chunk)
                    
                    if thread.is_alive():
                        logging.warning(f"Extraction timed out for chunk {i + chunk_idx} after {timeout_per_chunk} seconds")
                        continue
                    
                    if extraction_error:
                        logging.error(f"Error in chunk {i + chunk_idx}: {str(extraction_error)}")
                        continue
                        
                    if not response:
                        logging.warning(f"No response for chunk {i + chunk_idx}")
                        continue
                    
                    # Extract JSON from the response
                    try:
                        # Find the JSON part in the response
                        json_start = response.find('{')
                        json_end = response.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = response[json_start:json_end]
                            data = json.loads(json_str)
                            
                            # Add source metadata to each entity
                            for entity in data.get("entities", []):
                                entity["source"] = {
                                    "section_id": chunk.metadata.get("section_id"),
                                    "document_id": chunk.metadata.get("id"),
                                    "partner": chunk.metadata.get("partner"),
                                    "source": chunk.metadata.get("source")
                                }
                                
                                # Add these entities to our results
                                entities.append(entity)
                        else:
                            logging.warning(f"Could not find JSON in response for chunk {i + chunk_idx}")
                    except json.JSONDecodeError:
                        logging.error(f"Invalid JSON from LLM for chunk {i + chunk_idx}: {response}")
                except Exception as e:
                    logging.error(f"Error processing chunk {i + chunk_idx}: {str(e)}")
        
        logging.info(f"Extracted {len(entities)} entities from {total_chunks} chunks")
        return entities
    
    def save_processed_documents(
        self, 
        chunks: List[Document], 
        partner_name: Optional[str] = None
    ) -> str:
        """
        Save processed document chunks to disk.
        
        Args:
            chunks: List of document chunks
            partner_name: Partner name for the output directory
        
        Returns:
            Path to the saved file
        """
        output_dir = self.processed_data_dir
        if partner_name:
            output_dir = output_dir / partner_name
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert chunks to serializable format
        serializable_chunks = []
        for chunk in chunks:
            chunk_dict = {
                "content": chunk.page_content,
                "metadata": {k: v for k, v in chunk.metadata.items() if k != "embedding"}
            }
            
            # Handle embedding separately (convert numpy array to list)
            if "embedding" in chunk.metadata:
                embedding = chunk.metadata["embedding"]
                if isinstance(embedding, np.ndarray):
                    chunk_dict["embedding"] = embedding.tolist()
                else:
                    chunk_dict["embedding"] = embedding
            
            serializable_chunks.append(chunk_dict)
        
        # Save to JSON file
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        if partner_name:
            output_file = output_dir / f"{partner_name}_processed_{timestamp}.json"
        else:
            output_file = output_dir / f"processed_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({"chunks": serializable_chunks}, f)
        
        logger.info(f"Saved {len(chunks)} processed chunks to {output_file}")
        return str(output_file)
    
    def save_extracted_entities(
        self, 
        entities: List[Dict[str, Any]], 
        partner_name: Optional[str] = None
    ) -> str:
        """
        Save extracted entities to disk.
        
        Args:
            entities: List of extracted entities
            partner_name: Partner name for the output directory
        
        Returns:
            Path to the saved file
        """
        output_dir = self.processed_data_dir
        if partner_name:
            output_dir = output_dir / partner_name
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON file
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        if partner_name:
            output_file = output_dir / f"{partner_name}_entities_{timestamp}.json"
        else:
            output_file = output_dir / f"entities_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({"entities": entities}, f)
        
        logger.info(f"Saved {len(entities)} extracted entities to {output_file}")
        return str(output_file)
    
    def process_partner_documents(
        self, 
        partner_name: str,
        skip_entity_extraction: bool = False,
        entity_sample_limit: int = 100,
        entity_timeout: int = 60
    ) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """
        Process all documents for a partner.
        
        Args:
            partner_name: Name of the partner
            skip_entity_extraction: If True, skip the entity extraction step
            entity_sample_limit: Maximum number of chunks to process for entity extraction
            entity_timeout: Timeout in seconds per chunk for entity extraction
        
        Returns:
            Tuple of (processed document chunks, extracted entities)
        """
        logger.info(f"Processing documents for partner: {partner_name}")
        
        # Load documents
        documents = self.load_documents(partner_name=partner_name)
        if not documents:
            logger.warning(f"No documents found for partner: {partner_name}")
            return [], []
        
        # Split into chunks
        chunks = self.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks")
        
        # Generate embeddings
        chunks = self.generate_embeddings(chunks)
        
        # Save processed data first (so we don't lose work if entity extraction fails)
        self.save_processed_documents(chunks, partner_name)
        
        # Extract entities (optional)
        entities = []
        if not skip_entity_extraction:
            logger.info(f"Extracting entities (this may take a while for {len(chunks)} chunks)")
            try:
                # If there are too many chunks, limit processing time by sampling
                max_chunks = min(len(chunks), entity_sample_limit)
                if len(chunks) > max_chunks:
                    logger.warning(f"Limiting entity extraction to {max_chunks} chunks out of {len(chunks)} total chunks")
                    import random
                    random.seed(42)  # For reproducibility
                    chunks_for_entities = random.sample(chunks, max_chunks)
                else:
                    chunks_for_entities = chunks
                
                # Process with timeout
                entities = self.extract_entities(
                    chunks_for_entities, 
                    batch_size=5,
                    timeout_per_chunk=entity_timeout
                )
                
                # Save extracted entities
                self.save_extracted_entities(entities, partner_name)
            except Exception as e:
                logger.error(f"Error during entity extraction: {str(e)}")
                logger.info("Continuing with processing, but entity extraction failed")
        else:
            logger.info("Skipping entity extraction (--skip-entity-extraction flag set)")
        
        return chunks, entities 