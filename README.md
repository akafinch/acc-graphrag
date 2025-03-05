# GraphRAG Cloud Partner Onboarding System

A Graph-based Retrieval Augmented Generation (GraphRAG) system for onboarding cloud partners and enabling architects to quickly come up to speed with partner technologies.

## Overview

This system ingests partner documentation, builds a knowledge graph, and provides an intuitive interface for architects to query, explore, and learn about partner technologies through:

- Document ingestion and preprocessing
- Knowledge graph creation and enrichment
- Graph-aware RAG for accurate, contextual answers
- Interactive UI for knowledge exploration

## Technology Stack

- **Python**: Core programming language
- **Ollama**: Local LLM deployment for generation, embeddings, and analysis
- **Neo4j**: Graph database for storing interconnected knowledge
- **LangChain/LlamaIndex**: Framework for RAG pipelines
- **FastAPI**: Backend API
- **Streamlit**: Interactive user interface
- **Docker**: Containerization for easy deployment

## Project Structure

```
graphrag/
├── data/                  # Data storage
│   ├── raw/               # Raw partner documents
│   ├── processed/         # Processed documents
│   └── embeddings/        # Vector embeddings
├── ingestion/             # Document ingestion pipeline
├── database/              # Neo4j graph database interface
├── rag/                   # RAG implementation
├── api/                   # FastAPI backend
├── ui/                    # Streamlit interface
├── config/                # Configuration files
├── tests/                 # Unit and integration tests
└── docker/                # Dockerfiles and deployment
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/graphrag.git
cd graphrag

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start the services
docker-compose up -d
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- GPU with CUDA support (recommended for better LLM performance)
- 8GB+ RAM (16GB+ recommended)

### Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/graphrag.git
cd graphrag
```

2. **Start the services with Docker Compose**

```bash
docker-compose up -d
```

This will start:
- Neo4j graph database (accessible at http://localhost:7474)
- Ollama LLM service
- GraphRAG API
- Streamlit UI (accessible at http://localhost:8501)

3. **Set up Ollama models**

If the models aren't automatically pulled, you can pull them manually:

```bash
# Pull the LLM model for text generation
ollama pull llama3

# Pull the embedding model for vector embeddings
ollama pull nomic-embed-text
```

4. **Add partner documents**

Place partner documents in the appropriate directory:

```bash
mkdir -p data/raw/partner_name
cp /path/to/partner/documents/*.pdf data/raw/partner_name/
```

5. **Process documents and load to graph**

```bash
# When using Docker:
docker exec graphrag-api python -m ingestion.run [OPTIONS]

# When running directly:
python -m ingestion.run [OPTIONS]
```

#### Ingestion Command-line Options

| Option | Description |
|--------|-------------|
| `--partner PARTNER_NAME` | Name of the partner to process documents for |
| `--source-dir SOURCE_DIR` | Source directory for documents (overrides default) |
| `--load-to-graph` | Load processed data to the graph database |
| `--list-partners` | List partners with documents in the raw data directory |
| `--skip-entity-extraction` | Skip the entity extraction step (faster processing) |
| `--entity-limit LIMIT` | Maximum number of chunks to process for entity extraction (default: 100) |
| `--entity-timeout SECONDS` | Timeout in seconds per chunk for entity extraction (default: 60) |

#### Example Commands

```bash
# List available partners
python -m ingestion.run --list-partners

# Basic processing - process documents and load to graph
python -m ingestion.run --partner partner_name --load-to-graph

# Fast processing - skip the slow entity extraction step
python -m ingestion.run --partner partner_name --skip-entity-extraction --load-to-graph

# Optimized processing for large documents
python -m ingestion.run --partner partner_name --entity-limit 50 --entity-timeout 90 --load-to-graph

# Two-phase approach for large document sets:
# Phase 1: Process without entity extraction
python -m ingestion.run --partner partner_name --skip-entity-extraction --load-to-graph
# Phase 2: Add entities later from a sample of chunks
python -m ingestion.run --partner partner_name --entity-limit 50 --entity-timeout 90 --load-to-graph
```

6. **Access the UI**

Open your browser and navigate to:
- UI: http://localhost:8501
- API Documentation: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474 (user: neo4j, password: password)

### Running Without Docker

If you prefer to run the services directly:

1. **Start Neo4j**

Install Neo4j and the Graph Data Science plugin, and configure it to use the credentials in `.env`.

2. **Install Ollama**

Follow the instructions at [ollama.ai](https://ollama.ai) to install Ollama, then pull the required models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

3. **Set up Python environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Start the API and UI**

```bash
# Start the API
python -m api.main

# In another terminal
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
streamlit run ui/app.py
```

## Usage

### Adding Partner Documentation

1. Create a directory in `data/raw/` with the partner name
2. Add PDF, text, or other supported documents to this directory
3. Run the ingestion process with one of the following approaches:

#### Basic Approach (for small to medium document sets)
```bash
python -m ingestion.run --partner partner_name --load-to-graph
```

#### Fast Approach (for large document sets)
```bash
# Skip entity extraction for faster processing
python -m ingestion.run --partner partner_name --skip-entity-extraction --load-to-graph
```

#### Optimized Approach (for detailed entity extraction)
```bash
# Limit the number of chunks and set a timeout
python -m ingestion.run --partner partner_name --entity-limit 50 --entity-timeout 90 --load-to-graph
```

#### Two-Phase Approach (recommended for very large document sets)
```bash
# Phase 1: Process documents without entity extraction
python -m ingestion.run --partner partner_name --skip-entity-extraction --load-to-graph

# Phase 2: Extract entities from a sample of chunks
python -m ingestion.run --partner partner_name --entity-limit 50 --entity-timeout 90 --load-to-graph
```

The system supports various document formats including PDF, TXT, MD, and HTML, and will automatically process nested directories.

### Using the UI

The UI provides several ways to interact with the system:

- **Chat**: Ask questions about partner technologies and get informed answers
- **Search**: Find relevant information across partner documents and the knowledge graph
- **Entity Explorer**: Navigate through entities and their relationships
- **Graph Visualization**: Visualize the knowledge graph

### Using the API

The API provides the following endpoints:

- `/status`: Check the status of the system
- `/partners`: List available partners
- `/partners/process`: Process documents for a partner
- `/partners/load`: Load processed data to the graph
- `/search`: Search for relevant information
- `/entities/search`: Search for entities in the knowledge graph
- `/entities/{entity_id}`: Get detailed information about an entity
- `/chat`: Chat with the system using messages
- `/qa`: Get direct answers to questions

For full documentation, visit `http://localhost:8000/docs`

## Architecture

The GraphRAG system integrates the following components:

1. **Document Processing Pipeline**
   - Ingests various document formats
   - Chunks content for processing
   - Generates embeddings using Ollama
   - Extracts entities and relationships using LLM

2. **Knowledge Graph (Neo4j)**
   - Stores documents, sections, entities, and relationships
   - Enables graph-based queries and traversals
   - Provides context for question answering

3. **Graph-aware RAG**
   - Combines vector similarity search with graph traversal
   - Retrieves both document chunks and related entities
   - Follows relationship paths for multi-hop reasoning

4. **API Layer**
   - Handles document processing
   - Manages the graph database
   - Provides search, chat, and QA capabilities

5. **UI Layer**
   - Offers an intuitive interface for interaction
   - Visualizes knowledge graph connections
   - Provides multiple views into the data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 