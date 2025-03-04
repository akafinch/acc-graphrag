# GraphRAG Setup Guide

This guide provides detailed instructions for setting up and running the GraphRAG Cloud Partner Onboarding System.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Options](#installation-options)
  - [Docker-based Installation](#docker-based-installation)
  - [Direct Installation](#direct-installation)
- [Configuration](#configuration)
- [First-time Use](#first-time-use)
- [Adding Partner Documentation](#adding-partner-documentation)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## System Requirements

### Minimum Requirements

- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 10GB free space
- **Operating System**: Linux, macOS, or Windows

### Recommended Requirements

- **CPU**: 8+ cores
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU with CUDA support
- **Storage**: 20GB+ SSD
- **Operating System**: Linux (Ubuntu 20.04+)

## Installation Options

You can install GraphRAG using either Docker (recommended) or directly on your system.

### Docker-based Installation

This is the recommended approach as it handles all dependencies automatically.

#### Prerequisites

1. Install [Docker](https://docs.docker.com/get-docker/)
2. Install [Docker Compose](https://docs.docker.com/compose/install/)
3. (Optional, but recommended) Install NVIDIA Container Toolkit if you have a GPU:
   ```bash
   # For Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

#### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/graphrag.git
   cd graphrag
   ```

2. Create an environment file:
   ```bash
   cp .env.example .env
   ```
   
   You can edit the `.env` file if needed, but the default settings work for most cases.

3. Start the services with Docker Compose:
   ```bash
   docker-compose up -d
   ```

   This starts:
   - Neo4j graph database (accessible at http://localhost:7474)
   - Ollama LLM service
   - GraphRAG API
   - Streamlit UI (accessible at http://localhost:8501)

4. Pull required models for Ollama (if not automatically pulled):
   ```bash
   # Check if models are already available
   docker exec graphrag-ollama ollama list
   
   # If not, pull them
   docker exec graphrag-ollama ollama pull llama3
   docker exec graphrag-ollama ollama pull nomic-embed-text
   ```

5. Verify the installation by opening:
   - UI: http://localhost:8501
   - API Documentation: http://localhost:8000/docs
   - Neo4j Browser: http://localhost:7474 (user: neo4j, password: password)

### Direct Installation

If you prefer not to use Docker, you can install GraphRAG directly on your system.

#### Prerequisites

1. Install Python 3.10+:
   ```bash
   # For Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install python3.10 python3.10-venv python3.10-dev
   ```

2. Install Neo4j:
   - Download and install [Neo4j](https://neo4j.com/download/) (Community Edition)
   - Install the [Graph Data Science Plugin](https://neo4j.com/docs/graph-data-science/current/installation/)
   - Start Neo4j and create a new database named "graphrag"
   - Set the password to "password" (or update the .env file with your preferred password)

3. Install Ollama:
   - Follow the [installation instructions](https://ollama.ai/download) for your platform
   - Pull required models:
     ```bash
     ollama pull llama3
     ollama pull nomic-embed-text
     ```

#### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/graphrag.git
   cd graphrag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create an environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file to match your configuration, especially if your Neo4j is not running with default settings.

5. Start the API (in one terminal):
   ```bash
   python -m api.main
   ```

6. Start the UI (in another terminal):
   ```bash
   # Activate the virtual environment if not already activated
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Start the UI
   streamlit run ui/app.py
   ```

7. Verify the installation by opening:
   - UI: http://localhost:8501
   - API Documentation: http://localhost:8000/docs
   - Neo4j Browser: http://localhost:7474 (or your custom URL)

## Configuration

The system is configured using environment variables, which can be set in the `.env` file or directly in the environment.

### Key Configuration Options

```
# Ollama configuration
OLLAMA_HOST=http://localhost:11434  # Change if Ollama is running elsewhere
EMBEDDING_MODEL=nomic-embed-text     # Embedding model to use
LLM_MODEL=llama3                     # LLM model to use

# Neo4j configuration
NEO4J_URI=bolt://localhost:7687     # Neo4j connection URI
NEO4J_USER=neo4j                    # Neo4j username
NEO4J_PASSWORD=password             # Neo4j password
NEO4J_DATABASE=graphrag             # Neo4j database name

# API configuration
API_HOST=0.0.0.0                    # API host (0.0.0.0 to allow external connections)
API_PORT=8000                       # API port
API_WORKERS=4                       # Number of API workers

# UI configuration
UI_PORT=8501                        # UI port
```

For Docker installations, you typically don't need to change these settings as they're mapped correctly in the Docker Compose file.

## First-time Use

Once you have the system running, follow these steps to get started:

### 1. Verify System Status

1. Open the UI at http://localhost:8501
2. Navigate to the "About" page
3. Check that the system shows "Connected" for both Neo4j and Ollama services

### 2. Prepare a Test Document

1. Create a directory for a test partner:
   ```bash
   mkdir -p data/raw/test_partner
   ```

2. Add a sample PDF or text document to this directory. For testing, you can use any technical document:
   ```bash
   # Example: Copy a PDF file
   cp /path/to/sample_document.pdf data/raw/test_partner/
   ```

### 3. Process the Document

Process the document using the ingestion script:

```bash
# For Docker installation
docker exec graphrag-api python -m ingestion.run --partner test_partner --load-to-graph

# For direct installation
python -m ingestion.run --partner test_partner --load-to-graph
```

This command:
1. Loads documents from the `data/raw/test_partner` directory
2. Splits them into chunks
3. Generates embeddings
4. Extracts entities using the LLM
5. Loads everything into the Neo4j graph database

### 4. Explore the System

Once processing is complete, you can explore the system through the UI:

1. **Chat**: Ask questions about the processed documents
2. **Search**: Find specific information across documents
3. **Entity Explorer**: Browse the knowledge graph of extracted entities
4. **Graph Visualization**: Visualize relationships between entities

## Adding Partner Documentation

To add documentation for a new partner:

1. Create a partner directory:
   ```bash
   mkdir -p data/raw/partner_name
   ```

2. Add relevant documents to this directory:
   ```bash
   cp /path/to/partner/documents/*.pdf data/raw/partner_name/
   ```

3. Process the documents:
   ```bash
   # For Docker installation
   docker exec graphrag-api python -m ingestion.run --partner partner_name --load-to-graph

   # For direct installation
   python -m ingestion.run --partner partner_name --load-to-graph
   ```

4. Verify the processing:
   ```bash
   # List available partners
   python -m ingestion.run --list-partners
   ```

## Troubleshooting

### Common Issues

#### Neo4j Connection Issues

**Symptoms**: UI shows "Neo4j: Error" or API fails to start with Neo4j-related errors.

**Solutions**:
1. Check if Neo4j is running:
   ```bash
   # For Docker installation
   docker ps | grep neo4j
   
   # For direct installation
   sudo systemctl status neo4j  # Or equivalent for your system
   ```

2. Verify connection settings in `.env` file
3. Try connecting directly to Neo4j at http://localhost:7474

#### Ollama Issues

**Symptoms**: UI shows "Ollama: unavailable" or document processing fails with embedding errors.

**Solutions**:
1. Check if Ollama is running:
   ```bash
   # For Docker installation
   docker ps | grep ollama
   
   # For direct installation
   curl -s http://localhost:11434/api/tags | jq
   ```

2. Verify model availability:
   ```bash
   # For Docker installation
   docker exec graphrag-ollama ollama list
   
   # For direct installation
   ollama list
   ```

3. Pull required models if missing:
   ```bash
   # For Docker installation
   docker exec graphrag-ollama ollama pull llama3
   docker exec graphrag-ollama ollama pull nomic-embed-text
   
   # For direct installation
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

#### Document Processing Failures

**Symptoms**: Document processing fails or completes without errors but no entities appear in the system.

**Solutions**:
1. Check for errors in the logs:
   ```bash
   # For Docker installation
   docker logs graphrag-api
   
   # For direct installation
   # Check terminal output where you ran the ingestion command
   ```

2. Verify document format support:
   - The system supports PDF, TXT, MD, and HTML formats
   - Check that your documents are in a supported format
   - For PDFs, ensure they're text-based and not scanned images

3. Try processing with a smaller, simpler document to verify the pipeline works

#### UI Not Showing Results

**Symptoms**: UI runs but doesn't show any data or search results.

**Solutions**:
1. Verify the API is running:
   ```bash
   curl http://localhost:8000/status
   ```

2. Check API logs for errors:
   ```bash
   # For Docker installation
   docker logs graphrag-api
   
   # For direct installation
   # Check terminal where API is running
   ```

3. Verify document processing completed successfully

## Advanced Configuration

### Using Different LLM Models

You can use different models in Ollama:

1. Pull the desired model:
   ```bash
   ollama pull mistral  # or any other model
   ```

2. Update your `.env` file:
   ```
   LLM_MODEL=mistral
   ```

3. Restart the services:
   ```bash
   # For Docker installation
   docker-compose restart
   
   # For direct installation
   # Restart the API and UI processes
   ```

### Customizing Neo4j Schema

The graph schema is defined in `config/default.py`. You can customize node and relationship types by modifying this file and then restarting the system.

### Scaling for Larger Deployments

For larger deployments:

1. Increase Neo4j memory allocation in `docker-compose.yml`:
   ```yaml
   environment:
     - NEO4J_dbms_memory_pagecache_size=4G
     - NEO4J_dbms.memory.heap.initial_size=4G
     - NEO4J_dbms_memory_heap_max__size=8G
   ```

2. Increase API workers in `.env`:
   ```
   API_WORKERS=8
   ```

3. Consider using a more powerful GPU for Ollama 