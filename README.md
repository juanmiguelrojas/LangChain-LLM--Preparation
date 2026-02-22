# LangChain LLM & RAG Preparation

This repository contains two implementations built as part of the *Introduction to Creating RAGs with OpenAI* lab:

1. **Basic LangChain LLM Chain** – a simple question-answering chain using LangChain and OpenAI.
2. **RAG with Pinecone** – a full Retrieval-Augmented Generation pipeline using OpenAI embeddings and Pinecone as the vector database.

---

## Architecture

### Repository 1 – Basic LangChain LLM Chain (`basic_llm_chain.py`)

```
User Question
     │
     ▼
ChatPromptTemplate   ← system + human messages
     │
     ▼
ChatOpenAI (gpt-3.5-turbo)
     │
     ▼
StrOutputParser
     │
     ▼
Answer (string)
```

**Components:**
| Component | Description |
|---|---|
| `ChatOpenAI` | OpenAI GPT model wrapper |
| `ChatPromptTemplate` | Structures the system + user message |
| `StrOutputParser` | Extracts the plain-text response |
| LCEL `\|` pipe operator | Chains components into a single pipeline |

---

### Repository 2 – RAG with Pinecone (`rag_with_pinecone.py`)

```
Documents
    │
    ▼
RecursiveCharacterTextSplitter (chunks)
    │
    ▼
OpenAIEmbeddings (text-embedding-ada-002)
    │
    ▼
PineconeVectorStore  ←── stores & retrieves vectors
    │
    ▼ (at query time)
Retriever (top-k similarity search)
    │
    ▼
ChatPromptTemplate  ← context + question
    │
    ▼
ChatOpenAI (gpt-3.5-turbo)
    │
    ▼
StrOutputParser
    │
    ▼
Answer grounded in retrieved context
```

**Components:**
| Component | Role |
|---|---|
| `RecursiveCharacterTextSplitter` | Splits documents into 500-character chunks with 50-char overlap |
| `OpenAIEmbeddings` | Converts text chunks into 1536-dimensional vectors |
| `Pinecone` | Managed vector database – stores embeddings and runs similarity search |
| `PineconeVectorStore` | LangChain wrapper around the Pinecone index |
| `ChatOpenAI` | Generates the final answer given retrieved context |
| LCEL `\|` pipe operator | Wires retriever → prompt → LLM → parser |

---

## Prerequisites

- Python 3.10 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Pinecone API key](https://app.pinecone.io/) (free Starter tier works)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/juanmiguelrojas/LangChain-LLM--Preparation.git
cd LangChain-LLM--Preparation
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` in a text editor and set the following values:

```env
OPENAI_API_KEY=sk-...          # Your OpenAI API key
PINECONE_API_KEY=pcsk_...      # Your Pinecone API key
PINECONE_INDEX_NAME=langchain-rag-index   # Any name for your index
```

---

## Running the Code

### Basic LangChain LLM Chain

```bash
python basic_llm_chain.py
```

The script will:
1. Run three built-in example questions through the LLM chain and print the answers.
2. Start an interactive Q&A session – type any question and press Enter. Type `exit` to quit.

**Example output:**
```
============================================================
Basic LangChain LLM Chain Demo
============================================================

Question: What is LangChain and what is it used for?
----------------------------------------
Answer: LangChain is an open-source framework designed to simplify the
development of applications powered by large language models (LLMs)...
```

---

### RAG with Pinecone

```bash
python rag_with_pinecone.py
```

The script will:
1. **Ingest** five sample documents about LangChain, RAG, Pinecone, and OpenAI embeddings into your Pinecone index (only needed once; comment out the `ingest_documents` call on subsequent runs).
2. Run four example questions through the RAG chain, printing context-grounded answers.
3. Start an interactive Q&A session backed by the vector store.

**Example output:**
```
============================================================
RAG Demo – Questions answered from the knowledge base
============================================================

Question: How does RAG reduce hallucinations in LLMs?
----------------------------------------
Answer: RAG reduces hallucinations by first retrieving relevant documents
from an external knowledge base and passing those documents as context to
the language model. This grounds the model's output in factual information...
```

> **Note:** After the first successful run, open `rag_with_pinecone.py` and replace the `ingest_documents(...)` call with `load_vector_store(index_name)` to skip re-ingestion and avoid duplicate documents:
>
> ```python
> # vector_store = ingest_documents(SAMPLE_DOCUMENTS, index_name)  # first run only
> vector_store = load_vector_store(index_name)                      # subsequent runs
> ```

---

## Project Structure

```
LangChain-LLM--Preparation/
├── basic_llm_chain.py      # Repository 1 – Basic LLM chain demo
├── rag_with_pinecone.py    # Repository 2 – Full RAG pipeline
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
└── README.md               # This file
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` | Core LangChain framework |
| `langchain-openai` | OpenAI LLM and embeddings integration |
| `langchain-pinecone` | Pinecone vector store integration |
| `langchain-community` | Community integrations and text splitters |
| `pinecone-client` | Pinecone Python SDK |
| `openai` | Official OpenAI Python client |
| `python-dotenv` | Loads `.env` file into environment variables |
| `tiktoken` | Tokenizer used by OpenAI models |

---

## References

- [LangChain LLM Chain Tutorial](https://python.langchain.com/docs/tutorials/llm_chain/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain Pinecone Integration](https://python.langchain.com/docs/integrations/vectorstores/pinecone)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
