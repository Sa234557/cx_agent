"""
Vector store for FAQ retrieval.

We use ChromaDB (free, runs locally, no API needed) with 
sentence-transformers for embeddings (also free, runs locally).

This is the RAG (Retrieval Augmented Generation) component.
When a customer asks a question, we:
1. Embed the question into a vector
2. Search ChromaDB for similar FAQ vectors
3. Return the most relevant FAQ as context
"""

import chromadb
from chromadb.utils import embedding_functions
from data.knowledge_base import FAQ_DATA


# Use free local sentence-transformers model for embeddings
# This runs entirely on your machine, no API needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_vector_store():
    """Initialize ChromaDB with FAQ data."""
    
    # Create local ChromaDB client (stores data in memory for simplicity)
    client = chromadb.Client()
    
    # Use sentence-transformers for free local embeddings
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="faq_collection",
        embedding_function=embedding_fn
    )
    
    # Load FAQ data if collection is empty
    if collection.count() == 0:
        print("Loading FAQ data into vector store...")
        documents = []
        metadatas = []
        ids = []
        
        for i, faq in enumerate(FAQ_DATA):
            # Store question + answer as document
            documents.append(f"Question: {faq['question']}\nAnswer: {faq['answer']}")
            metadatas.append({"question": faq["question"]})
            ids.append(f"faq_{i}")
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Loaded {len(FAQ_DATA)} FAQs into vector store")
    
    return collection


def retrieve_context(query: str, collection, n_results: int = 2) -> tuple[str, float]:
    """
    Retrieve relevant FAQ context for a query.
    
    Returns:
        context: Relevant FAQ text
        confidence: Similarity score (0.0 to 1.0, higher = more similar)
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if not results["documents"][0]:
        return "", 0.0
    
    # ChromaDB returns distances (lower = more similar)
    # Convert to confidence score (higher = more confident)
    distance = results["distances"][0][0]
    confidence = max(0.0, 1.0 - distance)
    
    # Combine top results as context
    context = "\n\n".join(results["documents"][0])
    
    return context, confidence
