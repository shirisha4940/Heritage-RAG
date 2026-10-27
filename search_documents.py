
from pathlib import Path

from haystack import Pipeline
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


# Connect to the existing ChromaDB database
CHROMA_DIR = Path("chroma_db")

document_store = ChromaDocumentStore(
    collection_name="heritage_documents",
    persist_path=str(CHROMA_DIR.resolve()),
    distance_function="cosine"
)

# Convert the user's question into an embedding
text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Retrieve relevant document passages
retriever = ChromaEmbeddingRetriever(
    document_store=document_store,
    top_k=3
)

# Create the search pipeline
pipeline = Pipeline()

pipeline.add_component("text_embedder", text_embedder)
pipeline.add_component("retriever", retriever)

pipeline.connect(
    "text_embedder.embedding",
    "retriever.query_embedding"
)

# Ask a question
question = input("\nAsk a question about heritage restoration: ")

result = pipeline.run({
    "text_embedder": {
        "text": question
    }
})

documents = result["retriever"]["documents"]

print("\n--- Relevant information from your documents ---")

if not documents:
    print("No relevant passages found. Try another question.")
else:
    for i, document in enumerate(documents, start=1):
        print(f"\nResult {i}")
        print("Source:", document.meta.get("source", "Unknown"))
        print("Page:", document.meta.get("page_number", "Unknown"))
        print("Text:", document.content)
        print("-" * 50)
