"""
Neo4j client for the GraphRAG system.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from config.default import (NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI,
                           NEO4J_USER, NODE_TYPES, RELATIONSHIP_TYPES)

logger = logging.getLogger(__name__)

class Neo4jClient:
    """
    Client for interacting with the Neo4j graph database.
    """
    
    def __init__(
        self, 
        uri: str = NEO4J_URI, 
        user: str = NEO4J_USER, 
        password: str = NEO4J_PASSWORD,
        database: str = NEO4J_DATABASE
    ):
        """
        Initialize the Neo4j client with connection details.
        """
        self.uri = uri
        self.user = user
        self.password = password
        # Always use 'neo4j' database in Community Edition
        self.database = "neo4j"
        self.driver = None
        
        logger.info(f"Initialized Neo4j client with URI: {uri}, Database: {self.database}")
        
    def connect(self) -> None:
        """Connect to the Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # Test the connection
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j database at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close the connection to the Neo4j database."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def setup_database(self) -> None:
        """
        Set up the database schema with constraints and indexes.
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Check if we can connect
                result = session.run("RETURN 1 as test").single()
                if result and result["test"] == 1:
                    logger.info(f"Connected to database '{self.database}'")
                    
                    # Create constraints for each node type
                    for node_type in NODE_TYPES:
                        constraint_name = f"{node_type.lower()}_id_constraint"
                        query = f"""
                        CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                        FOR (n:{node_type})
                        REQUIRE n.id IS UNIQUE
                        """
                        session.run(query)
                        logger.info(f"Created constraint: {constraint_name}")
                    
                    # Try to create simple indexes for searchable properties
                    # This is more compatible across Neo4j versions
                    for node_type in NODE_TYPES:
                        # Create standard indexes for common properties
                        for prop in ["name", "title", "content", "description"]:
                            if prop in NODE_TYPES[node_type]["properties"]:
                                try:
                                    index_name = f"{node_type.lower()}_{prop}_index"
                                    query = f"""
                                    CREATE INDEX {index_name} IF NOT EXISTS 
                                    FOR (n:{node_type}) 
                                    ON (n.{prop})
                                    """
                                    session.run(query)
                                    logger.info(f"Created index: {index_name}")
                                except Exception as e:
                                    logger.warning(f"Could not create index on {node_type}.{prop}: {str(e)}")
        except Exception as e:
            logger.error(f"Error setting up database schema: {str(e)}")
    
    def run_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return the results.
        
        Args:
            query: The Cypher query to execute
            parameters: Parameters for the query
        
        Returns:
            List of dictionaries with query results
        """
        if not self.driver:
            self.connect()
            
        parameters = parameters or {}
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Neo4jError as e:
            logger.error(f"Neo4j query error: {str(e)}")
            raise
    
    def create_node(
        self, 
        node_type: str, 
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create or update a node in the graph database.
        
        Args:
            node_type: Type of node to create
            properties: Node properties
        
        Returns:
            Created or updated node data
        """
        if node_type not in NODE_TYPES:
            raise ValueError(f"Invalid node type: {node_type}")
        
        # Use MERGE instead of CREATE to handle existing nodes
        query = f"""
        MERGE (n:{node_type} {{id: $properties.id}})
        SET n += $properties
        RETURN n
        """
        try:
            result = self.run_query(query, {"properties": properties})
            return result[0]["n"] if result else None
        except Exception as e:
            logger.error(f"Error creating/updating {node_type} node: {str(e)}")
            raise
    
    def create_relationship(
        self, 
        from_node_id: str, 
        to_node_id: str, 
        relationship_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create or update a relationship between two nodes.
        
        Args:
            from_node_id: ID of the source node
            to_node_id: ID of the target node
            relationship_type: Type of relationship
            properties: Relationship properties
        
        Returns:
            Created or updated relationship data
        """
        if relationship_type not in RELATIONSHIP_TYPES:
            raise ValueError(f"Invalid relationship type: {relationship_type}")
        
        properties = properties or {}
        
        query = f"""
        MATCH (a), (b)
        WHERE a.id = $from_id AND b.id = $to_id
        MERGE (a)-[r:{relationship_type}]->(b)
        SET r += $properties
        RETURN a, r, b
        """
        
        try:
            result = self.run_query(query, {
                "from_id": from_node_id,
                "to_id": to_node_id,
                "properties": properties
            })
            
            return result[0] if result else None
        except Exception as e:
            logger.warning(f"Error creating relationship from {from_node_id} to {to_node_id}: {str(e)}")
            return None
    
    def get_node_by_id(self, node_id: str) -> Dict[str, Any]:
        """
        Get a node by its ID.
        
        Args:
            node_id: ID of the node to retrieve
        
        Returns:
            Node data or None if not found
        """
        query = """
        MATCH (n)
        WHERE n.id = $id
        RETURN n
        """
        result = self.run_query(query, {"id": node_id})
        return result[0]["n"] if result else None
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """
        Get all nodes of a specific type.
        
        Args:
            node_type: Type of nodes to retrieve
        
        Returns:
            List of node data
        """
        if node_type not in NODE_TYPES:
            raise ValueError(f"Invalid node type: {node_type}")
        
        query = f"""
        MATCH (n:{node_type})
        RETURN n
        """
        result = self.run_query(query)
        return [record["n"] for record in result]
    
    def search_nodes(
        self, 
        search_text: str, 
        node_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for nodes containing specific text in their properties.
        
        Args:
            search_text: Text to search for
            node_types: Types of nodes to search (if None, search all)
        
        Returns:
            List of matching nodes
        """
        node_types = node_types or list(NODE_TYPES.keys())
        invalid_types = [t for t in node_types if t not in NODE_TYPES]
        if invalid_types:
            raise ValueError(f"Invalid node types: {', '.join(invalid_types)}")
        
        # Build a UNION query for each node type
        queries = []
        for node_type in node_types:
            query = f"""
            CALL db.index.fulltext.queryNodes('{node_type.lower()}_name_index', $search_text) 
            YIELD node, score
            RETURN node, score, '{node_type}' as type
            """
            queries.append(query)
            
            if "description" in NODE_TYPES[node_type]["properties"]:
                query = f"""
                CALL db.index.fulltext.queryNodes('{node_type.lower()}_description_index', $search_text) 
                YIELD node, score
                RETURN node, score, '{node_type}' as type
                """
                queries.append(query)
        
        full_query = " UNION ".join(queries) + " ORDER BY score DESC LIMIT 20"
        
        return self.run_query(full_query, {"search_text": search_text})
    
    def get_node_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all relationships for a specific node.
        
        Args:
            node_id: ID of the node
        
        Returns:
            List of relationship data
        """
        query = """
        MATCH (n)-[r]-(m)
        WHERE n.id = $id
        RETURN n, r, m
        """
        return self.run_query(query, {"id": node_id})
    
    def get_path_between_nodes(
        self, 
        from_node_id: str, 
        to_node_id: str, 
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find paths between two nodes.
        
        Args:
            from_node_id: Start node ID
            to_node_id: End node ID
            max_depth: Maximum path length
        
        Returns:
            List of paths
        """
        query = """
        MATCH path = shortestPath((a)-[*1..{max_depth}]-(b))
        WHERE a.id = $from_id AND b.id = $to_id
        RETURN path
        """.format(max_depth=max_depth)
        
        parameters = {
            "from_id": from_node_id,
            "to_id": to_node_id
        }
        
        return self.run_query(query, parameters)
    
    def get_vector_search_results(
        self, 
        embedding: List[float], 
        node_type: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform a vector similarity search on nodes with embeddings.
        
        Args:
            embedding: Query embedding vector
            node_type: Type of nodes to search
            top_k: Number of results to return
        
        Returns:
            List of similar nodes with similarity scores
        """
        if node_type not in NODE_TYPES:
            raise ValueError(f"Invalid node type: {node_type}")
        
        if "embedding" not in NODE_TYPES[node_type]["properties"]:
            raise ValueError(f"Node type {node_type} does not have embedding property")
        
        # Using Neo4j's vector similarity functions
        query = f"""
        MATCH (n:{node_type})
        WHERE n.embedding IS NOT NULL
        WITH n, gds.similarity.cosine(n.embedding, $embedding) AS similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        RETURN n, similarity
        """
        
        parameters = {
            "embedding": embedding,
            "top_k": top_k
        }
        
        return self.run_query(query, parameters)
    
    def get_subgraph(
        self, 
        center_node_id: str, 
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get a subgraph centered around a specific node.
        
        Args:
            center_node_id: ID of the central node
            depth: Number of relationship hops to include
        
        Returns:
            Subgraph data including nodes and relationships
        """
        query = """
        MATCH path = (n)-[*0..{depth}]-(m)
        WHERE n.id = $center_id
        RETURN path
        """.format(depth=depth)
        
        result = self.run_query(query, {"center_id": center_node_id})
        
        # Process the result to extract unique nodes and relationships
        nodes = {}
        relationships = []
        
        for record in result:
            path = record.get("path", [])
            for segment in path:
                if hasattr(segment, "start") and hasattr(segment, "end"):
                    # This is a relationship
                    rel_data = {
                        "id": segment.id,
                        "type": segment.type,
                        "source": segment.start_node.id,
                        "target": segment.end_node.id,
                        "properties": dict(segment)
                    }
                    relationships.append(rel_data)
                else:
                    # This is a node
                    node_id = segment.id
                    if node_id not in nodes:
                        nodes[node_id] = {
                            "id": node_id,
                            "labels": list(segment.labels),
                            "properties": dict(segment)
                        }
        
        return {
            "nodes": list(nodes.values()),
            "relationships": relationships
        } 