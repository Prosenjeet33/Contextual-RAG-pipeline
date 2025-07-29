# Contextual RAG Chatbot in Open WebUI

## Overview

This project implements an interactive Chatbot within Open WebUI, powered by a Contextual RAG (Retrieval Augmented Generation) pipeline. It utilizes the following technologies:

- **Document Processing** with Docling
- **Data Pipeline & Storage**: Docling for ingestion, PGVector/PostgreSQL for embedding storage
- **Retrieval-Augmented Generation (RAG)**: LlamaIndex + PGVector
- **Contextual RAG**: Anthropic-style embedding, LLM, and re-ranking models
- **Local LLMs**: Ollama for hosting agentic models
- **Prompt Optimization**: Crew.AI
- **Evaluation Playground**: Arize Phoenix Prompt Playground
- **LLMOps**: RAGAs for tracing & debugging
- **Chatbot Interface**: Arize Phoenix integrated into Open WebUI

---

## Architecture

```mermaid
graph TD
    A[Open WebUI Chat Interface] -->|User Query| B[Contextual RAG Pipeline]
    B -->|Docling Ingestion| C[Docling Data Pipeline]
    C -->|Vector Embedding| D[PGVector/PostgreSQL]
    B -->|Retrieve| D
    B -->|LLM (Ollama)| E[Local Model]
    B -->|Re-ranking| F[Anthropic-style Embedding/Model]
    B -->|Prompt Optimization| G[Crew.AI]
    B -->|Tracing & Debug| H[RAGAs]
    B -->|Evaluation| I[Arize Phoenix Prompt Playground]
    B -->|Response| A
```

---

## Implementation Steps

### 1. Document Processing

- **Docling**: Ingest, preprocess, and chunk documents.  
- **Pipeline**: Store document chunks and metadata.

### 2. Embedding & Storage

- **LlamaIndex**: Generate embeddings for document chunks.
- **PGVector/PostgreSQL**: Store embeddings and metadata for fast retrieval.

### 3. Contextual RAG Pipeline

- **Retrieve**: Use LlamaIndex and PGVector to fetch relevant chunks.
- **Re-rank**: Apply Anthropic-style contextual embedding for improved relevance.
- **LLM**: Use Ollama-hosted models for local inference.

### 4. Agentic & Prompt Optimization

- **Crew.AI**: Optimize prompts and manage agentic flows.
- **Arize Phoenix Prompt Playground**: Evaluate prompt effectiveness and chatbot responses.

### 5. LLMOps: Tracing & Debugging

- **RAGAs**: Integrate for traceability, debugging, and performance monitoring.

### 6. Chatbot Interface

- **Arize Phoenix**: Integrate the interface into Open WebUI for seamless user experience.

---

## Example Pipeline (Python)

```python name=app/main.py
import docling
from llama_index import VectorStoreIndex, ServiceContext
from pgvector import PGVectorStore
from ollama import OllamaLLM
from crewai import optimize_prompt
from arize_phoenix import PromptPlayground
import ragas

# 1. Ingest and chunk documents
docs = docling.ingest("data/")
chunks = docling.chunk(docs)

# 2. Embedding and Storage
service_ctx = ServiceContext.from_defaults()
index = VectorStoreIndex.from_documents(chunks, vector_store=PGVectorStore(...), service_context=service_ctx)

# 3. RAG Retrieval
query = "User's question"
retrieved_chunks = index.query(query)

# 4. Re-ranking (Anthropic-style)
reranked = rerank_with_anthropic_embedding(retrieved_chunks, query)

# 5. Local LLM (Ollama)
llm = OllamaLLM(model="your-model")
response = llm.generate_response(reranked, query)

# 6. Prompt Optimization
optimized_prompt = optimize_prompt(query, context=reranked)

# 7. Evaluation Playground
PromptPlayground.evaluate(query, response)

# 8. Tracing & Debugging
ragas.trace(query, response)

# 9. Serve via Open WebUI with Arize Phoenix Chatbot
```

---

## Deployment

- **Open WebUI**: Host the chatbot, connect backend pipeline.
- **Ollama**: Ensure local model serving.
- **PGVector/PostgreSQL**: Database setup for embeddings.
- **Arize Phoenix**: Integrate for chatbot interface and prompt evaluation.

---

## Next Steps

1. Setup PostgreSQL + PGVector.
2. Configure Docling for document ingestion.
3. Integrate LlamaIndex for embeddings and retrieval.
4. Deploy Ollama for local LLM inference.
5. Implement re-ranking with Anthropic-style embedding/model.
6. Integrate Crew.AI, Arize Phoenix, RAGAs.
7. Connect all components within Open WebUI.

---

## References

- [Docling](https://github.com/docling)
- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [PGVector](https://github.com/pgvector/pgvector)
- [Ollama](https://ollama.com/)
- [Crew.AI](https://github.com/crewai/crewai)
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)
- [RAGAs](https://github.com/explodinggradients/ragas)