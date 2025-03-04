"""
Graph-aware retriever for the GraphRAG system.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from langchain.docstore.document import Document

from config.default import TOP_K_RETRIEVAL
from database.neo4j_client import Neo4jClient
from rag.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class GraphRetriever:
    """
    Graph-aware retriever that combines vector search with graph-based retrieval.
    """
    
    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        ollama_client: Optional[OllamaClient] = None,
        top_k: int = TOP_K_RETRIEVAL
    ):
        """Initialize the graph retriever."""
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.ollama_client = ollama_client or OllamaClient()
        self.top_k = top_k
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        node_types: Optional[List[str]] = None,
        partners: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents and entities for a query.
        
        Args:
            query: Query string
            top_k: Number of results to retrieve (default: self.top_k)
            node_types: Types of nodes to search (if None, search all types)
            partners: Filter by partner names (if None, search all partners)
        
        Returns:
            List of retrieved items with relevance scores
        """
        top_k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.ollama_client.generate_embedding(query)
        
        # Perform vector search for sections
        section_results = self._vector_search_sections(
            query_embedding, 
            top_k=top_k, 
            partners=partners
        )
        
        # Search for related entities
        entity_results = self._search_entities(
            query, 
            query_embedding, 
            top_k=top_k, 
            node_types=node_types,
            partners=partners
        )
        
        # Combine results with appropriate metadata
        combined_results = section_results + entity_results
        
        # Sort by relevance score
        combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Apply graph expansion to top results
        expanded_results = self._expand_results_with_graph(combined_results[:top_k])
        
        return expanded_results
    
    def _vector_search_sections(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        partners: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on section nodes.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to retrieve
            partners: Filter by partner names
        
        Returns:
            List of section results with scores
        """
        # Construct Cypher query for vector search with optional partner filter
        query_parts = [
            "MATCH (s:Section)",
            "WHERE s.embedding IS NOT NULL"
        ]
        
        if partners:
            partner_list = ', '.join([f"'{p}'" for p in partners])
            query_parts.append(f"AND s.partner IN [{partner_list}]")
        
        query_parts.extend([
            "WITH s, gds.similarity.cosine(s.embedding, $embedding) AS score",
            "WHERE score > 0.7",  # Similarity threshold
            "RETURN s, score",
            "ORDER BY score DESC",
            f"LIMIT {top_k}"
        ])
        
        # Execute query
        cypher_query = "\n".join(query_parts)
        results = self.neo4j_client.run_query(
            cypher_query, 
            {"embedding": query_embedding}
        )
        
        # Format results
        formatted_results = []
        for record in results:
            section = record.get("s", {})
            score = record.get("score", 0)
            
            formatted_results.append({
                "id": section.get("id"),
                "type": "Section",
                "content": section.get("content"),
                "metadata": {
                    "source": section.get("source"),
                    "document_id": section.get("document_id"),
                    "partner": section.get("partner"),
                    "section_id": section.get("section_id"),
                    "section_number": section.get("section_number")
                },
                "score": score
            })
        
        return formatted_results
    
    def _search_entities(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        node_types: Optional[List[str]] = None,
        partners: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for entities by combining vector similarity and text search.
        
        Args:
            query: Text query
            query_embedding: Query embedding vector
            top_k: Number of results to retrieve
            node_types: Types of entities to search
            partners: Filter by partner names
        
        Returns:
            List of entity results with scores
        """
        # Default node types if not specified
        if not node_types:
            node_types = ["Concept", "Product", "Service", "Technology", "UseCase"]
        
        # Combine results from vector similarity and text search
        vector_results = []
        for node_type in node_types:
            try:
                # Vector similarity search
                vector_query = """
                MATCH (n:{node_type})
                WHERE n.embedding IS NOT NULL
                {partner_filter}
                WITH n, gds.similarity.cosine(n.embedding, $embedding) AS score
                WHERE score > 0.7
                RETURN n, score, '{node_type}' as type
                ORDER BY score DESC
                LIMIT $top_k
                """.format(
                    node_type=node_type,
                    partner_filter=f"AND n.partner IN $partners" if partners else ""
                )
                
                params = {
                    "embedding": query_embedding,
                    "top_k": top_k
                }
                if partners:
                    params["partners"] = partners
                
                results = self.neo4j_client.run_query(vector_query, params)
                vector_results.extend(results)
            except Exception as e:
                logger.error(f"Error in vector search for {node_type}: {str(e)}")
        
        # Text search using fulltext indexes
        text_results = []
        try:
            # Build a UNION query for text search on each node type
            text_queries = []
            for node_type in node_types:
                # Search in name
                query_part = f"""
                CALL db.index.fulltext.queryNodes('{node_type.lower()}_name_index', $query)
                YIELD node, score
                WHERE score > 0.3
                {f"AND node.partner IN $partners" if partners else ""}
                RETURN node, score, '{node_type}' as type
                """
                text_queries.append(query_part)
                
                # Search in description if applicable
                query_part = f"""
                CALL db.index.fulltext.queryNodes('{node_type.lower()}_description_index', $query)
                YIELD node, score
                WHERE score > 0.3
                {f"AND node.partner IN $partners" if partners else ""}
                RETURN node, score, '{node_type}' as type
                """
                text_queries.append(query_part)
            
            # Execute combined query
            combined_query = " UNION ".join(text_queries) + " ORDER BY score DESC LIMIT $top_k"
            params = {"query": query, "top_k": top_k}
            if partners:
                params["partners"] = partners
                
            text_results = self.neo4j_client.run_query(combined_query, params)
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
        
        # Combine and deduplicate results
        all_results = vector_results + text_results
        deduplicated_results = {}
        
        for record in all_results:
            node = record.get("node", {})
            node_id = node.get("id")
            
            if node_id not in deduplicated_results:
                deduplicated_results[node_id] = {
                    "id": node_id,
                    "type": record.get("type"),
                    "name": node.get("name"),
                    "description": node.get("description"),
                    "metadata": {
                        "partner": node.get("partner"),
                        "category": node.get("category", None),
                        "version": node.get("version", None)
                    },
                    "score": record.get("score", 0)
                }
            else:
                # Keep the higher score
                current_score = deduplicated_results[node_id]["score"]
                new_score = record.get("score", 0)
                if new_score > current_score:
                    deduplicated_results[node_id]["score"] = new_score
        
        # Convert to list and sort by score
        results = list(deduplicated_results.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to top_k
        return results[:top_k]
    
    def _expand_results_with_graph(
        self,
        results: List[Dict[str, Any]],
        depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Expand retrieval results with graph relationships.
        
        Args:
            results: Initial retrieval results
            depth: Depth of graph expansion
        
        Returns:
            Expanded results with graph context
        """
        expanded_results = []
        
        for result in results:
            result_id = result.get("id")
            if not result_id:
                expanded_results.append(result)
                continue
            
            # Get the subgraph for this result
            try:
                subgraph = self.neo4j_client.get_subgraph(result_id, depth=depth)
                
                # Add related nodes and relationships to result
                result["graph_context"] = {
                    "nodes": subgraph.get("nodes", []),
                    "relationships": subgraph.get("relationships", [])
                }
                
                expanded_results.append(result)
            except Exception as e:
                logger.error(f"Error expanding result with graph: {str(e)}")
                expanded_results.append(result)
        
        return expanded_results
    
    def get_graph_context(
        self,
        node_id: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get detailed graph context for a specific node.
        
        Args:
            node_id: ID of the node
            depth: Depth of graph traversal
        
        Returns:
            Dictionary with graph context information
        """
        # Get subgraph
        subgraph = self.neo4j_client.get_subgraph(node_id, depth=depth)
        
        # Get information about the central node
        central_node = self.neo4j_client.get_node_by_id(node_id)
        
        # Get different types of relationships
        related_items = {}
        relationships = subgraph.get("relationships", [])
        
        for rel in relationships:
            rel_type = rel.get("type")
            source_id = rel.get("source")
            target_id = rel.get("target")
            
            # Only care about relationships where our node is the source
            if source_id == node_id:
                if rel_type not in related_items:
                    related_items[rel_type] = []
                
                # Find the target node in the subgraph
                target_node = None
                for node in subgraph.get("nodes", []):
                    if node.get("id") == target_id:
                        target_node = node
                        break
                
                if target_node:
                    related_items[rel_type].append(target_node)
        
        return {
            "central_node": central_node,
            "related_items": related_items,
            "subgraph": subgraph
        }
    
    def retrieve_for_qa(
        self,
        question: str,
        top_k: Optional[int] = None,
        partners: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Specialized retrieval for question answering.
        
        Args:
            question: Question string
            top_k: Number of results to retrieve
            partners: Filter by partner names
        
        Returns:
            Tuple of (retrieval results, graph context)
        """
        # Retrieve initial results
        results = self.retrieve(
            query=question,
            top_k=top_k,
            partners=partners
        )
        
        # Structure information for the LLM
        context_docs = []
        graph_context = []
        
        for result in results:
            result_type = result.get("type")
            
            if result_type == "Section":
                # Format document content
                section_content = result.get("content", "")
                metadata = result.get("metadata", {})
                source = metadata.get("source", "Unknown source")
                partner = metadata.get("partner", "Unknown partner")
                
                context_docs.append({
                    "content": section_content,
                    "source": source,
                    "partner": partner,
                    "score": result.get("score", 0)
                })
            else:
                # Format entity information with graph context
                entity_name = result.get("name", "")
                entity_description = result.get("description", "")
                entity_type = result.get("type", "")
                
                # Include graph context
                graph_info = result.get("graph_context", {})
                relationships = []
                
                # Extract relationship information
                for rel in graph_info.get("relationships", []):
                    rel_type = rel.get("type")
                    source_id = rel.get("source")
                    target_id = rel.get("target")
                    
                    # Find source and target nodes
                    source_node = None
                    target_node = None
                    for node in graph_info.get("nodes", []):
                        if node.get("id") == source_id:
                            source_node = node
                        if node.get("id") == target_id:
                            target_node = node
                        if source_node and target_node:
                            break
                    
                    if source_node and target_node:
                        relationships.append({
                            "relation": rel_type,
                            "source": {
                                "id": source_id,
                                "name": source_node.get("properties", {}).get("name", ""),
                                "type": next(iter(source_node.get("labels", [])), "")
                            },
                            "target": {
                                "id": target_id,
                                "name": target_node.get("properties", {}).get("name", ""),
                                "type": next(iter(target_node.get("labels", [])), "")
                            }
                        })
                
                graph_context.append({
                    "entity": {
                        "name": entity_name,
                        "type": entity_type,
                        "description": entity_description
                    },
                    "relationships": relationships,
                    "score": result.get("score", 0)
                })
        
        return (context_docs, graph_context) 