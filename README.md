# Mentis — KI (AI) Challenge
Digital memory: a personal knowledge retrieval system using Retrieval-Augmented Generation (RAG).
Note: This is the public version of the Mentis backend. The private development history has been omitted; release snapshots are published here.
## Quick Start

```bash
make install
make chat
```

## Setup

### 1. Environment Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.exampe .env
```

Edit `.env` with your API keys:
- `OPENAI_API_KEY`: Your OpenAI API key
- `WEAVIATE_URL` and `WEAVIATE_API_KEY`: Provided via email to Fiona Koenz

### 2. Install Dependencies

```bash
make install
```

## Usage

### Available Commands

- `make install` - Install project dependencies
- `make eval` - Run retrieval evaluation
- `make chat` - Start chat interface with all 3 retrievers
- `make mentis` - Start Mentis chat interface (main retriever only)
- `make encode-rags` - Encode data for SimpleRag and SummaryRag
- `make encode-main` - Encode data using main encoder
- `make clean` - Clean temporary files

### Chat Interfaces

**Standard Chat (All Retrievers)**:
```bash
make chat
```
Launches the full chat interface with access to all three retrieval systems.

**Mentis Chat (Main Retriever Only)**:
```bash
make mentis
```
Launches a focused chat interface using only the main retriever system.

## Requirements

- Python 3.12+
- OpenAI API access
- Weaviate database access

### Dependencies

- `openai` - OpenAI API client
- `pydantic` - Data validation
- `weaviate-client` - Vector database client
- `ragas` - Evaluation framework

## Features

- Vector-based knowledge retrieval
- Summary-based retrieval system  
- Automated evaluation using LLM assessment
- Multiple chat interfaces for different use cases

## Important Note for KI Challenge Evaluators

⚠️ **API Keys Notice**: The Weaviate API keys required for this project have been sent to Fiona Koenz via email. If you encounter authentication errors when running the evaluation, please use the provided keys or contact me (Levin) for assistance.
