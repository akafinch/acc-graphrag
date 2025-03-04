"""
FastAPI backend for the GraphRAG system.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from config.default import API_HOST, API_PORT, API_WORKERS
from database.neo4j_client import Neo4jClient
from ingestion.document_processor import DocumentProcessor
from rag.graph_retriever import GraphRetriever
from rag.ollama_client import OllamaClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize API
app = FastAPI(
    title="GraphRAG API",
    description="API for the Graph-based Retrieval Augmented Generation system",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
neo4j_client = Neo4jClient()
ollama_client = OllamaClient()
document_processor = DocumentProcessor(ollama_client=ollama_client)
graph_retriever = GraphRetriever(
    neo4j_client=neo4j_client,
    ollama_client=ollama_client
)

# Data models
class PartnerDocuments(BaseModel):
    partner_name: str
    documents_path: Optional[str] = None

class Query(BaseModel):
    text: str
    top_k: Optional[int] = 5
    partners: Optional[List[str]] = None
    node_types: Optional[List[str]] = None

class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    partners: Optional[List[str]] = None

class EntitySearch(BaseModel):
    query: str
    node_types: Optional[List[str]] = None
    partners: Optional[List[str]] = None
    top_k: Optional[int] = 10

# API routes
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "GraphRAG API"}

@app.get("/status")
async def status():
    """Get service status."""
    try:
        # Check Neo4j connection
        neo4j_status = "ok"
        try:
            neo4j_client.connect()
        except Exception as e:
            neo4j_status = str(e)
        
        # Check Ollama service
        ollama_status = "ok" if ollama_client.check_status() else "unavailable"
        ollama_models = ollama_client.list_models()
        
        return {
            "status": "ok",
            "services": {
                "neo4j": neo4j_status,
                "ollama": ollama_status
            },
            "models": ollama_models
        }
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/partners/process")
async def process_partner_documents(request: PartnerDocuments):
    """
    Process documents for a partner.
    
    This will:
    1. Load documents from the partner directory
    2. Split into chunks
    3. Generate embeddings
    4. Extract entities
    5. Save processed data
    """
    try:
        partner_name = request.partner_name
        
        # Process documents
        chunks, entities = document_processor.process_partner_documents(partner_name)
        
        return {
            "status": "success",
            "partner": partner_name,
            "processed_documents": len(chunks),
            "extracted_entities": len(entities)
        }
    except Exception as e:
        logger.error(f"Error processing partner documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/partners/load")
async def load_partner_to_graph(request: PartnerDocuments):
    """
    Load processed partner data into the graph database.
    
    This will:
    1. Load processed documents and entities from disk
    2. Create nodes and relationships in the graph
    """
    try:
        partner_name = request.partner_name
        
        # TODO: Implement graph loading logic
        # This would load the processed chunks and entities and create nodes/relationships
        
        return {
            "status": "success",
            "partner": partner_name,
            "message": "Not implemented yet"
        }
    except Exception as e:
        logger.error(f"Error loading partner to graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/partners")
async def list_partners():
    """List all partners in the system."""
    try:
        # Query Neo4j for unique partner names
        query = """
        MATCH (n)
        WHERE n.partner IS NOT NULL
        RETURN DISTINCT n.partner AS partner
        """
        
        results = neo4j_client.run_query(query)
        partners = [record["partner"] for record in results if "partner" in record]
        
        return {
            "partners": partners
        }
    except Exception as e:
        logger.error(f"Error listing partners: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: Query):
    """
    Search for relevant documents and entities.
    """
    try:
        results = graph_retriever.retrieve(
            query=request.text,
            top_k=request.top_k,
            node_types=request.node_types,
            partners=request.partners
        )
        
        return {
            "query": request.text,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/entities/search")
async def search_entities(request: EntitySearch):
    """
    Search for entities in the knowledge graph.
    """
    try:
        # Generate query embedding
        query_embedding = ollama_client.generate_embedding(request.query)
        
        # Search for entities
        results = graph_retriever._search_entities(
            query=request.query,
            query_embedding=query_embedding,
            top_k=request.top_k,
            node_types=request.node_types,
            partners=request.partners
        )
        
        return {
            "query": request.query,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error searching entities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/entities/{entity_id}")
async def get_entity(entity_id: str, depth: int = 2):
    """
    Get detailed information about an entity with its graph context.
    """
    try:
        context = graph_retriever.get_graph_context(entity_id, depth=depth)
        
        return context
    except Exception as e:
        logger.error(f"Error getting entity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat with the GraphRAG system.
    """
    try:
        # Extract the last user message
        last_user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                last_user_message = msg.content
                break
        
        if not last_user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Retrieve relevant context
        context_docs, graph_context = graph_retriever.retrieve_for_qa(
            question=last_user_message,
            partners=request.partners
        )
        
        # Format context for the prompt
        formatted_context = []
        
        # Add document chunks
        for i, doc in enumerate(context_docs):
            formatted_context.append(f"Document {i+1} (Source: {doc['source']}, Partner: {doc['partner']}):\n{doc['content']}")
        
        # Add graph context
        for i, entity in enumerate(graph_context):
            entity_info = entity["entity"]
            relationships = entity["relationships"]
            
            context_str = f"Entity {i+1}: {entity_info['name']} (Type: {entity_info['type']})\n"
            context_str += f"Description: {entity_info['description']}\n"
            
            if relationships:
                context_str += "Relationships:\n"
                for rel in relationships:
                    src = rel["source"]["name"]
                    rel_type = rel["relation"]
                    tgt = rel["target"]["name"]
                    context_str += f"- {src} {rel_type} {tgt}\n"
            
            formatted_context.append(context_str)
        
        # Combine context
        context_text = "\n\n".join(formatted_context)
        
        # Create a system message with context
        system_prompt = f"""You are an AI assistant for a cloud architecture team. 
Use the following retrieved information to answer the user's question accurately.
If the information doesn't contain the answer, say you don't know rather than making up information.

Context information:
{context_text}
"""
        
        # Create messages for chat completion
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add previous conversation messages, but skip system messages
        for msg in request.messages:
            if msg.role != "system":
                messages.append({"role": msg.role, "content": msg.content})
        
        # Handle streaming response
        if request.stream:
            return StreamingResponse(
                stream_chat_response(messages, request.temperature),
                media_type="text/event-stream"
            )
        else:
            # Get chat completion
            response = ollama_client.chat_completion(
                messages=messages,
                temperature=request.temperature
            )
            
            return {
                "response": response,
                "context": {
                    "documents": context_docs,
                    "graph_entities": graph_context
                }
            }
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def stream_chat_response(messages, temperature):
    """Stream chat response in SSE format."""
    try:
        stream = ollama_client.chat_completion(
            messages=messages,
            temperature=temperature,
            stream=True
        )
        
        response_text = ""
        for chunk in ollama_client.process_stream(stream):
            if chunk:
                response_text += chunk
                yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                time.sleep(0.01)  # Small delay to avoid flooding
        
        # Send completion message
        yield f"data: {json.dumps({'text': response_text, 'done': True})}\n\n"
    except Exception as e:
        logger.error(f"Error in stream_chat_response: {str(e)}")
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

@app.post("/qa")
async def question_answering(request: Query):
    """
    Get an answer to a question with supporting context.
    """
    try:
        # Retrieve relevant context
        context_docs, graph_context = graph_retriever.retrieve_for_qa(
            question=request.text,
            partners=request.partners
        )
        
        # Format context for the prompt
        formatted_context = []
        
        # Add document chunks
        for i, doc in enumerate(context_docs):
            formatted_context.append(f"Document {i+1} (Source: {doc['source']}, Partner: {doc['partner']}):\n{doc['content']}")
        
        # Add graph context
        for i, entity in enumerate(graph_context):
            entity_info = entity["entity"]
            relationships = entity["relationships"]
            
            context_str = f"Entity {i+1}: {entity_info['name']} (Type: {entity_info['type']})\n"
            context_str += f"Description: {entity_info['description']}\n"
            
            if relationships:
                context_str += "Relationships:\n"
                for rel in relationships:
                    src = rel["source"]["name"]
                    rel_type = rel["relation"]
                    tgt = rel["target"]["name"]
                    context_str += f"- {src} {rel_type} {tgt}\n"
            
            formatted_context.append(context_str)
        
        # Combine context
        context_text = "\n\n".join(formatted_context)
        
        # Create prompt
        prompt = f"""
Question: {request.text}

Context information:
{context_text}

Please provide a comprehensive answer to the question based on the provided context.
If the information doesn't contain the answer, please say you don't know rather than making up information.
        """
        
        # Get answer from LLM
        answer = ollama_client.generate_text(prompt=prompt, temperature=0.7)
        
        return {
            "question": request.text,
            "answer": answer,
            "context": {
                "documents": context_docs,
                "graph_entities": graph_context
            }
        }
    except Exception as e:
        logger.error(f"Error in QA: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        workers=API_WORKERS
    ) 