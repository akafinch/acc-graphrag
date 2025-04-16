"""
Streamlit UI for the GraphRAG system.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_chat import message

from config.default import API_HOST, API_PORT, PAGE_ICON, PAGE_TITLE, UI_PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API endpoint
API_URL = f"http://{API_HOST}:{API_PORT}"

# Page config
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .api-status-good {
        color: green;
        font-weight: bold;
    }
    .api-status-bad {
        color: red;
        font-weight: bold;
    }
    .entity-card {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .entity-name {
        font-weight: bold;
        font-size: 16px;
    }
    .entity-type {
        font-size: 12px;
        color: #666;
        margin-bottom: 5px;
    }
    .entity-description {
        font-size: 14px;
        margin-bottom: 10px;
    }
    .relationship {
        font-size: 12px;
        color: #333;
        padding: 2px 5px;
        background-color: #e0e0e0;
        border-radius: 3px;
        margin-right: 5px;
        display: inline-block;
    }
    .document-section {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        border-left: 4px solid #2c7fb8;
    }
    .document-source {
        font-size: 12px;
        color: #666;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions
def check_api_status() -> Tuple[bool, Dict]:
    """Check the status of the GraphRAG API."""
    try:
        response = httpx.get(f"{API_URL}/status", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return False, {"error": f"Connection Error: {str(e)}"}

def get_partners() -> List[str]:
    """Get list of available partners."""
    try:
        response = httpx.get(f"{API_URL}/partners", timeout=5)
        if response.status_code == 200:
            return response.json().get("partners", [])
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting partners: {str(e)}")
        return []

def search(query: str, top_k: int = 5, partners: Optional[List[str]] = None) -> Dict:
    """Search for information relevant to a query."""
    try:
        data = {
            "text": query,
            "top_k": top_k
        }
        
        if partners:
            data["partners"] = partners
        
        response = httpx.post(f"{API_URL}/search", json=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        return {"error": f"Error: {str(e)}"}

def search_entities(query: str, node_types: Optional[List[str]] = None, partners: Optional[List[str]] = None, top_k: int = 10) -> Dict:
    """Search for entities in the knowledge graph."""
    try:
        data = {
            "query": query,
            "top_k": top_k
        }
        
        if node_types:
            data["node_types"] = node_types
            
        if partners:
            data["partners"] = partners
        
        response = httpx.post(f"{API_URL}/entities/search", json=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error searching entities: {str(e)}")
        return {"error": f"Error: {str(e)}"}

def get_entity_details(entity_id: str, depth: int = 2) -> Dict:
    """Get detailed information about an entity with its graph context."""
    try:
        response = httpx.get(f"{API_URL}/entities/{entity_id}?depth={depth}", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error getting entity details: {str(e)}")
        return {"error": f"Error: {str(e)}"}

def ask_question(question: str, partners: Optional[List[str]] = None) -> Dict:
    """Ask a question to the GraphRAG system."""
    try:
        data = {
            "text": question
        }
        
        if partners:
            data["partners"] = partners
        
        response = httpx.post(f"{API_URL}/qa", json=data, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error asking question: {str(e)}")
        return {"error": f"Error: {str(e)}"}

def chat(messages: List[Dict], stream: bool = False, partners: Optional[List[str]] = None) -> Dict:
    """Chat with the GraphRAG system."""
    try:
        data = {
            "messages": messages,
            "stream": stream
        }
        
        if partners:
            data["partners"] = partners
        
        if stream:
            # Return the response object for streaming
            return httpx.post(f"{API_URL}/chat", json=data, timeout=60, stream=True)
        else:
            response = httpx.post(f"{API_URL}/chat", json=data, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error chatting: {str(e)}")
        return {"error": f"Error: {str(e)}"}

def visualize_graph(nodes_data, relationships_data):
    """Visualize a knowledge graph using streamlit-agraph."""
    nodes = []
    edges = []
    
    # Create nodes
    for node in nodes_data:
        label = node.get("properties", {}).get("name", "Unknown")
        node_id = node.get("id", "")
        node_type = next(iter(node.get("labels", [])), "Unknown")
        
        # Assign color based on node type
        color = "#636EFA"  # Default blue
        if node_type == "Concept":
            color = "#636EFA"  # Blue
        elif node_type == "Product":
            color = "#EF553B"  # Red
        elif node_type == "Service":
            color = "#00CC96"  # Green
        elif node_type == "Technology":
            color = "#AB63FA"  # Purple
        elif node_type == "UseCase":
            color = "#FFA15A"  # Orange
        elif node_type == "Section":
            color = "#19D3F3"  # Cyan
        elif node_type == "Document":
            color = "#FF6692"  # Pink
        
        nodes.append(Node(id=node_id, 
                         label=label, 
                         size=20, 
                         color=color,
                         title=f"{node_type}: {label}"))
    
    # Create edges
    for rel in relationships_data:
        source = rel.get("source", "")
        target = rel.get("target", "")
        rel_type = rel.get("type", "")
        
        edges.append(Edge(source=source, 
                         target=target, 
                         label=rel_type))
    
    # Graph configuration
    config = Config(width=700,
                   height=500,
                   directed=True,
                   physics=True,
                   hierarchical=False)
    
    return agraph(nodes=nodes, edges=edges, config=config)

# App layout
def main():
    """Main application."""
    # Sidebar
    st.sidebar.title("Cloud Partner Knowledge Explorer")
    
    # Check API status
    api_status, status_data = check_api_status()
    
    if api_status:
        st.sidebar.markdown(f"<p class='api-status-good'>API Status: Connected</p>", unsafe_allow_html=True)
        
        # Show service status
        services = status_data.get("services", {})
        for service, status in services.items():
            if status == "ok":
                st.sidebar.markdown(f"<p>{service.title()}: <span class='api-status-good'>OK</span></p>", unsafe_allow_html=True)
            else:
                st.sidebar.markdown(f"<p>{service.title()}: <span class='api-status-bad'>Error</span></p>", unsafe_allow_html=True)
                st.sidebar.text(status)
        
        # Available models
        models = status_data.get("models", [])
        if models:
            st.sidebar.subheader("Available Models")
            for model in models:
                st.sidebar.text(f"• {model}")
    else:
        st.sidebar.markdown(f"<p class='api-status-bad'>API Status: Disconnected</p>", unsafe_allow_html=True)
        st.sidebar.text(status_data.get("error", "Unknown error"))
        st.error("Cannot connect to the GraphRAG API. Please check that the API is running.")
        return
    
    # Get partners
    partners = get_partners()
    
    # Partner selection
    selected_partners = st.sidebar.multiselect(
        "Filter by Partners",
        options=partners,
        default=None
    )
    
    # Navigation
    nav_options = ["Chat", "Search", "Entity Explorer", "Graph Visualization", "About"]
    selected_nav = st.sidebar.radio("Navigation", nav_options)
    
    # Main content
    if selected_nav == "Chat":
        chat_page(selected_partners)
    elif selected_nav == "Search":
        search_page(selected_partners)
    elif selected_nav == "Entity Explorer":
        entity_explorer_page(selected_partners)
    elif selected_nav == "Graph Visualization":
        graph_visualization_page(selected_partners)
    else:
        about_page()

def chat_page(selected_partners: List[str]):
    """Chat interface for interacting with the GraphRAG system."""
    st.title("Chat with the Knowledge Explorer")
    st.write("Ask questions about partner technologies and get informed answers with context.")
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Format for API
    def format_messages(messages):
        return [{"role": "user" if i % 2 == 0 else "assistant", "content": msg} for i, msg in enumerate(messages)]
    
    # Display chat messages
    for i, msg in enumerate(st.session_state.chat_messages):
        message(msg, is_user=(i % 2 == 0), key=f"msg_{i}")
    
    # Chat input
    user_input = st.text_input("Your question:", key="chat_input")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_messages.append(user_input)
        
        # Format messages for API
        api_messages = format_messages(st.session_state.chat_messages)
        
        # Call API
        with st.spinner("Thinking..."):
            result = chat(api_messages, stream=False, partners=selected_partners)
        
        if "error" in result:
            st.error(result["error"])
        else:
            # Get response
            assistant_response = result.get("response", "I'm sorry, I couldn't process your request.")
            
            # Add assistant response to chat
            st.session_state.chat_messages.append(assistant_response)
            
            # Show context
            with st.expander("View Supporting Context"):
                # Document context
                docs = result.get("context", {}).get("documents", [])
                if docs:
                    st.subheader("Document Sources")
                    for i, doc in enumerate(docs):
                        source = doc.get("source", "Unknown")
                        partner = doc.get("partner", "Unknown")
                        content = doc.get("content", "")
                        
                        st.markdown(f"""
                        <div class="document-section">
                            <div class="document-source">Source: {source} | Partner: {partner}</div>
                            <div>{content}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Graph context
                entities = result.get("context", {}).get("graph_entities", [])
                if entities:
                    st.subheader("Knowledge Graph Context")
                    for entity in entities:
                        entity_info = entity.get("entity", {})
                        name = entity_info.get("name", "Unknown")
                        entity_type = entity_info.get("type", "Unknown")
                        description = entity_info.get("description", "")
                        
                        st.markdown(f"""
                        <div class="entity-card">
                            <div class="entity-name">{name}</div>
                            <div class="entity-type">{entity_type}</div>
                            <div class="entity-description">{description}</div>
                        """, unsafe_allow_html=True)
                        
                        # Show relationships
                        relationships = entity.get("relationships", [])
                        if relationships:
                            for rel in relationships:
                                source = rel.get("source", {}).get("name", "")
                                rel_type = rel.get("relation", "")
                                target = rel.get("target", {}).get("name", "")
                                
                                st.markdown(f"""
                                <div class="relationship">{source} → {rel_type} → {target}</div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
        
        # Reset input
        st.text_input("Your question:", value="", key="chat_input_reset")
    
    # Clear chat button
    if st.button("Clear Chat") and "chat_messages" in st.session_state:
        st.session_state.chat_messages = []
        st.experimental_rerun()

def search_page(selected_partners: List[str]):
    """Search interface for finding information in documents and knowledge graph."""
    st.title("Search Partner Knowledge")
    st.write("Find relevant information across partner documentation and knowledge graph.")
    
    # Search input
    query = st.text_input("Search query:")
    top_k = st.slider("Number of results", min_value=3, max_value=20, value=5)
    
    if query:
        with st.spinner("Searching..."):
            results = search(query, top_k=top_k, partners=selected_partners)
        
        if "error" in results:
            st.error(results["error"])
        else:
            st.subheader(f"Results for: {query}")
            
            # Display results
            results_data = results.get("results", [])
            if not results_data:
                st.info("No results found.")
            else:
                # Group results by type
                sections = []
                entities = []
                
                for result in results_data:
                    result_type = result.get("type", "")
                    if result_type == "Section":
                        sections.append(result)
                    else:
                        entities.append(result)
                
                # Create tabs for different result types
                tab1, tab2 = st.tabs(["Document Sections", "Knowledge Graph Entities"])
                
                with tab1:
                    if sections:
                        for section in sections:
                            content = section.get("content", "")
                            metadata = section.get("metadata", {})
                            source = metadata.get("source", "Unknown")
                            partner = metadata.get("partner", "Unknown")
                            score = section.get("score", 0)
                            
                            st.markdown(f"""
                            <div class="document-section">
                                <div class="document-source">Source: {source} | Partner: {partner} | Relevance: {score:.2f}</div>
                                <div>{content}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No document sections found.")
                
                with tab2:
                    if entities:
                        for entity in entities:
                            name = entity.get("name", "Unknown")
                            entity_type = entity.get("type", "Unknown")
                            description = entity.get("description", "")
                            metadata = entity.get("metadata", {})
                            partner = metadata.get("partner", "Unknown")
                            score = entity.get("score", 0)
                            entity_id = entity.get("id", "")
                            
                            st.markdown(f"""
                            <div class="entity-card">
                                <div class="entity-name">{name}</div>
                                <div class="entity-type">{entity_type} | Partner: {partner} | Relevance: {score:.2f}</div>
                                <div class="entity-description">{description}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Button to view entity details
                            if st.button(f"View Details: {name}", key=f"btn_{entity_id}"):
                                st.session_state.selected_entity = entity_id
                                st.session_state.selected_nav = "Entity Explorer"
                                st.experimental_rerun()
                    else:
                        st.info("No knowledge graph entities found.")

def entity_explorer_page(selected_partners: List[str]):
    """Explore entities in the knowledge graph."""
    st.title("Entity Explorer")
    st.write("Explore entities in the knowledge graph and their relationships.")
    
    # Entity search
    query = st.text_input("Search for entities:")
    
    # Entity types multiselect
    entity_types = ["Concept", "Product", "Service", "Technology", "UseCase"]
    selected_types = st.multiselect("Entity Types", entity_types, default=entity_types)
    
    if query:
        with st.spinner("Searching..."):
            results = search_entities(
                query, 
                node_types=selected_types,
                partners=selected_partners
            )
        
        if "error" in results:
            st.error(results["error"])
        else:
            st.subheader(f"Results for: {query}")
            
            # Display entity search results
            entities = results.get("results", [])
            if not entities:
                st.info("No entities found.")
            else:
                # Create a df for better display
                entity_data = []
                for entity in entities:
                    entity_data.append({
                        "Name": entity.get("name", "Unknown"),
                        "Type": entity.get("type", "Unknown"),
                        "Description": entity.get("description", "")[:100] + "...",
                        "Partner": entity.get("metadata", {}).get("partner", "Unknown"),
                        "Relevance": f"{entity.get('score', 0):.2f}",
                        "ID": entity.get("id", "")
                    })
                
                df = pd.DataFrame(entity_data)
                st.dataframe(df)
                
                # Allow selecting an entity to view details
                selected_entity = st.selectbox(
                    "Select an entity to view details:",
                    options=df["ID"].tolist(),
                    format_func=lambda x: df[df["ID"] == x]["Name"].iloc[0]
                )
                
                if selected_entity:
                    view_entity_details(selected_entity)
    
    # View entity details if selected in another page
    if "selected_entity" in st.session_state:
        entity_id = st.session_state.selected_entity
        view_entity_details(entity_id)
        # Clear the selection after viewing
        del st.session_state.selected_entity

def view_entity_details(entity_id: str):
    """View detailed information about an entity."""
    with st.spinner("Loading entity details..."):
        details = get_entity_details(entity_id)
    
    if "error" in details:
        st.error(details["error"])
        return
    
    # Extract entity information
    central_node = details.get("central_node", {})
    node_properties = central_node.get("properties", {})
    node_name = node_properties.get("name", "Unknown")
    node_type = next(iter(central_node.get("labels", [])), "Unknown")
    node_description = node_properties.get("description", "")
    
    # Display entity information
    st.header(node_name)
    st.subheader(f"Type: {node_type}")
    st.write(node_description)
    
    # Display related items
    related_items = details.get("related_items", {})
    if related_items:
        st.subheader("Relationships")
        
        for rel_type, items in related_items.items():
            st.write(f"**{rel_type}**")
            
            for item in items:
                item_props = item.get("properties", {})
                item_name = item_props.get("name", "Unknown")
                item_type = next(iter(item.get("labels", [])), "Unknown")
                item_description = item_props.get("description", "")[:100]
                if item_description:
                    item_description += "..."
                
                st.markdown(f"""
                <div class="entity-card" style="margin-left: 20px;">
                    <div class="entity-name">{item_name}</div>
                    <div class="entity-type">{item_type}</div>
                    <div class="entity-description">{item_description}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Visualize subgraph
    st.subheader("Knowledge Graph Visualization")
    subgraph = details.get("subgraph", {})
    nodes = subgraph.get("nodes", [])
    relationships = subgraph.get("relationships", [])
    
    visualize_graph(nodes, relationships)

def graph_visualization_page(selected_partners: List[str]):
    """Visualize the knowledge graph."""
    st.title("Knowledge Graph Visualization")
    st.write("Explore the connections between entities in the knowledge graph.")
    
    # Entity search for central node
    query = st.text_input("Search for a central entity:")
    
    if query:
        with st.spinner("Searching..."):
            results = search_entities(
                query, 
                partners=selected_partners,
                top_k=5
            )
        
        if "error" in results:
            st.error(results["error"])
        else:
            # Create options for selecting an entity
            entities = results.get("results", [])
            if not entities:
                st.info("No entities found.")
            else:
                # Format options for selection
                entity_options = {f"{e['name']} ({e['type']})": e["id"] for e in entities}
                
                selected_entity_name = st.selectbox(
                    "Select a central entity for the graph:",
                    options=list(entity_options.keys())
                )
                
                # Graph depth
                depth = st.slider("Graph depth", min_value=1, max_value=3, value=2)
                
                if selected_entity_name:
                    selected_entity_id = entity_options[selected_entity_name]
                    
                    with st.spinner("Loading graph..."):
                        details = get_entity_details(selected_entity_id, depth=depth)
                    
                    if "error" in details:
                        st.error(details["error"])
                    else:
                        # Visualize the graph
                        subgraph = details.get("subgraph", {})
                        nodes = subgraph.get("nodes", [])
                        relationships = subgraph.get("relationships", [])
                        
                        st.subheader(f"Knowledge Graph for {selected_entity_name}")
                        
                        # Show stats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Nodes", len(nodes))
                        with col2:
                            st.metric("Relationships", len(relationships))
                        
                        visualize_graph(nodes, relationships)
                        
                        # Show node list
                        with st.expander("View Node List"):
                            node_data = []
                            for node in nodes:
                                node_data.append({
                                    "Name": node.get("properties", {}).get("name", "Unknown"),
                                    "Type": next(iter(node.get("labels", [])), "Unknown"),
                                    "ID": node.get("id", "")
                                })
                            
                            st.dataframe(pd.DataFrame(node_data))

def about_page():
    """About page with information about the GraphRAG system."""
    st.title("About Cloud Partner Knowledge Explorer")
    
    st.markdown("""
    ## GraphRAG: Graph-based Retrieval Augmented Generation

    The Cloud Partner Knowledge Explorer is a system that helps architects quickly learn about and build with partner technologies by:

    1. **Ingesting partner documentation** - Process and analyze partner documentation to extract knowledge
    2. **Building a knowledge graph** - Create interconnected entities representing concepts, products, services, and technologies
    3. **Providing accurate answers** - Use graph-aware RAG to retrieve relevant context and generate accurate responses
    4. **Enabling knowledge exploration** - Visualize the knowledge graph to discover connections between technologies

    ### Technology Stack

    - **Python**: Core programming language
    - **Ollama**: Local LLM deployment for generation, embeddings, and analysis
    - **Neo4j**: Graph database for storing interconnected knowledge
    - **LangChain/LlamaIndex**: Framework for RAG pipelines
    - **FastAPI**: Backend API
    - **Streamlit**: Interactive user interface
    
    ### Benefits

    - **Contextual understanding** - Graph structure captures relationships between technologies
    - **Multi-hop reasoning** - Follow paths through the graph to answer complex questions
    - **Knowledge discovery** - Explore connections that weren't explicitly stated in documents
    - **Structured responses** - Information is organized by entity types and relationships
    """)

if __name__ == "__main__":
    main() 