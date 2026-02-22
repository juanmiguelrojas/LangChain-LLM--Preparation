"""
RAG (Retrieval-Augmented Generation) with LangChain, OpenAI, and Pinecone
==========================================================================
This script demonstrates how to build a RAG pipeline using:
- OpenAI for embeddings and the language model
- Pinecone as the vector database
- LangChain as the orchestration framework

Based on:
  - https://python.langchain.com/docs/tutorials/rag/
  - https://python.langchain.com/docs/integrations/vectorstores/pinecone

Architecture:
  1. Document ingestion  -> chunk text into smaller pieces
  2. Embedding           -> convert chunks to vectors using OpenAI
  3. Vector storage      -> store vectors in Pinecone
  4. Retrieval           -> find relevant chunks for a query
  5. Generation          -> pass context + query to OpenAI LLM for an answer
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500       # Maximum number of characters per document chunk
CHUNK_OVERLAP = 50     # Number of overlapping characters between consecutive chunks
TOP_K_RESULTS = 3      # Number of most-relevant chunks to retrieve per query

# ---------------------------------------------------------------------------
# Sample documents to ingest into the vector store
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENTS = [
    Document(
        page_content=(
            "LangChain is a framework for developing applications powered by large language "
            "models (LLMs). It provides tools for chaining together components such as prompts, "
            "models, memory, and agents. LangChain simplifies complex workflows that involve "
            "sequential model calls, data retrieval, and decision-making."
        ),
        metadata={"source": "langchain_intro", "topic": "LangChain"},
    ),
    Document(
        page_content=(
            "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by "
            "first retrieving relevant documents from an external knowledge base, then passing "
            "those documents as context to the language model. This grounds the model's output "
            "in factual, up-to-date information and significantly reduces hallucination."
        ),
        metadata={"source": "rag_overview", "topic": "RAG"},
    ),
    Document(
        page_content=(
            "Pinecone is a managed vector database designed for storing and querying high-dimensional "
            "embeddings at scale. It supports approximate nearest-neighbor search and integrates "
            "natively with popular ML frameworks. Pinecone handles index management, replication, "
            "and scaling automatically, making it ideal for production RAG applications."
        ),
        metadata={"source": "pinecone_overview", "topic": "Pinecone"},
    ),
    Document(
        page_content=(
            "OpenAI embeddings convert text into numerical vectors that capture semantic meaning. "
            "The text-embedding-ada-002 model produces 1536-dimensional vectors. These embeddings "
            "allow similarity searches: texts with similar meaning have vectors that are close "
            "together in the vector space, enabling effective document retrieval."
        ),
        metadata={"source": "openai_embeddings", "topic": "Embeddings"},
    ),
    Document(
        page_content=(
            "A vector database stores embeddings (numerical representations of data) and enables "
            "fast similarity search. When a user submits a query, the query is also embedded and "
            "the database returns the most semantically similar documents. This is the retrieval "
            "step in a RAG pipeline."
        ),
        metadata={"source": "vector_db_overview", "topic": "Vector Database"},
    ),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_pinecone_index(index_name: str, dimension: int = 1536):
    """
    Returns an existing Pinecone index or creates a new one.

    Args:
        index_name: Name of the Pinecone index.
        dimension:  Dimensionality of the embeddings (1536 for text-embedding-ada-002).

    Returns:
        A Pinecone Index object.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Using existing Pinecone index '{index_name}'.")

    return pc.Index(index_name)


def ingest_documents(documents: list[Document], index_name: str) -> PineconeVectorStore:
    """
    Splits documents into chunks, embeds them, and stores them in Pinecone.

    Args:
        documents:  List of LangChain Document objects.
        index_name: Name of the Pinecone index to use.

    Returns:
        A PineconeVectorStore instance ready for retrieval.
    """
    # Split documents into smaller chunks for better retrieval granularity
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Create embeddings model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Ensure the index exists
    get_pinecone_index(index_name)

    # Store the chunks in Pinecone
    print("Storing embeddings in Pinecone...")
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    print("Documents successfully ingested into Pinecone.")
    return vector_store


def load_vector_store(index_name: str) -> PineconeVectorStore:
    """
    Loads an existing Pinecone vector store without re-ingesting documents.

    Args:
        index_name: Name of the Pinecone index.

    Returns:
        A PineconeVectorStore instance.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )


def build_rag_chain(vector_store: PineconeVectorStore):
    """
    Builds the RAG chain: retriever -> prompt -> LLM -> output parser.

    The chain:
      1. Retrieves relevant document chunks from Pinecone based on the question.
      2. Formats a prompt with the retrieved context and the question.
      3. Sends the prompt to OpenAI and returns the answer.

    Args:
        vector_store: A PineconeVectorStore to use for retrieval.

    Returns:
        A LangChain LCEL chain.
    """
    # Create a retriever that returns the top-3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # RAG prompt: instructs the model to use only the provided context
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant. Answer the user's question using ONLY "
                    "the information provided in the context below. If the answer is not "
                    "in the context, say 'I don't have enough information to answer that.'\n\n"
                    "Context:\n{context}"
                ),
            ),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        """Concatenate retrieved document chunks into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Build the chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    # Validate required environment variables
    missing = [
        var
        for var in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME")
        if not os.getenv(var)
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Please copy .env.example to .env and fill in your credentials."
        )

    index_name = os.getenv("PINECONE_INDEX_NAME")

    # Ingest sample documents (comment out after first run to avoid duplicates)
    vector_store = ingest_documents(SAMPLE_DOCUMENTS, index_name)

    # Build the RAG chain
    rag_chain = build_rag_chain(vector_store)

    # Run example queries
    example_questions = [
        "What is LangChain and what can I use it for?",
        "How does RAG reduce hallucinations in LLMs?",
        "What makes Pinecone suitable for production RAG applications?",
        "How do OpenAI embeddings represent text semantically?",
    ]

    print("\n" + "=" * 60)
    print("RAG Demo – Questions answered from the knowledge base")
    print("=" * 60)

    for question in example_questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        answer = rag_chain.invoke(question)
        print(f"Answer: {answer}")

    # Interactive session
    print("\n" + "=" * 60)
    print("Interactive RAG Session (type 'exit' to quit)")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not question:
            print("Please enter a question.")
            continue
        answer = rag_chain.invoke(question)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
