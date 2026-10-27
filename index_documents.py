
from pathlib import Path
import hashlib
import pymupdf

from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


# Project folders
DOCUMENTS_DIR = Path("documents")
CHROMA_DIR = Path("chroma_db")

CHROMA_DIR.mkdir(exist_ok=True)

print("Starting Heritage-RAG document indexing...")


# Step 1: Read PDF files
documents = []

pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found inside the documents folder."
    )

for pdf_file in pdf_files:
    print(f"Reading: {pdf_file.name}")

    with pymupdf.open(pdf_file) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if text:
                documents.append(
                    Document(
                        content=text,
                        meta={
                            "source": pdf_file.name,
                            "page_number": page_number
                        }
                    )
                )

if not documents:
    raise ValueError(
        "No readable text found. The PDFs may be scanned images."
    )

print(f"Read {len(documents)} PDF pages.")


# Step 2: Split text into smaller chunks
splitter = DocumentSplitter(
    split_by="word",
    split_length=250,
    split_overlap=40
)

chunks = splitter.run(documents=documents)["documents"]

# Give each chunk a stable ID so rerunning the script
# does not create a new ID for the same source passage.
for chunk in chunks:
    identity = (
        str(chunk.meta.get("source", ""))
        + "|"
        + str(chunk.meta.get("page_number", ""))
        + "|"
        + str(chunk.meta.get("split_id", ""))
        + "|"
        + str(chunk.content)
    )
    chunk.id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

print(f"Created {len(chunks)} text chunks.")


# Step 3: Generate document embeddings
embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

embedder.warm_up()

embedded_chunks = embedder.run(
    documents=chunks
)["documents"]

print("Document embeddings created.")


# Step 4: Save documents and embeddings in ChromaDB
document_store = ChromaDocumentStore(
    collection_name="heritage_documents",
    persist_path=str(CHROMA_DIR.resolve()),
    distance_function="cosine"
)

document_store.write_documents(embedded_chunks)

print("Documents saved in ChromaDB.")
print(f"Total indexed chunks: {len(embedded_chunks)}")
print("Heritage-RAG indexing completed successfully!")
