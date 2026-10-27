import hashlib
import os
import re
from pathlib import Path

import pymupdf
from fasthtml.common import *
from starlette.datastructures import UploadFile

from haystack import Document, Pipeline
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.retrievers.chroma import (
    ChromaEmbeddingRetriever,
)
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from transformers import pipeline


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "heritage_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

TOP_K = 3
MAX_PDF_SIZE = 15 * 1024 * 1024
CHUNK_WORDS = 180
CHUNK_OVERLAP = 35

CHROMA_PATH.mkdir(parents=True, exist_ok=True)


# ==================================================
# HAYSTACK AND CHROMADB SETUP
# ==================================================

document_store = ChromaDocumentStore(
    collection_name=COLLECTION_NAME,
    persist_path=str(CHROMA_PATH),
    distance_function="cosine",
)

query_embedder = SentenceTransformersTextEmbedder(
    model=EMBEDDING_MODEL
)
query_embedder.warm_up()

document_embedder = SentenceTransformersDocumentEmbedder(
    model=EMBEDDING_MODEL
)
document_embedder.warm_up()

retrieval_pipeline = Pipeline()

retrieval_pipeline.add_component("query_embedder", query_embedder)
retrieval_pipeline.add_component(
    "retriever",
    ChromaEmbeddingRetriever(document_store=document_store),
)

retrieval_pipeline.connect(
    "query_embedder.embedding",
    "retriever.query_embedding",
)

generator = pipeline(
    "text-generation",
    model=GENERATION_MODEL,
    tokenizer=GENERATION_MODEL,
)


# ==================================================
# PDF CHUNKING AND INDEXING
# ==================================================

def split_into_chunks(
    text,
    chunk_words=CHUNK_WORDS,
    overlap=CHUNK_OVERLAP,
):
    words = text.split()

    if not words:
        return []

    chunks = []
    step = max(1, chunk_words - overlap)

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words])

        if chunk.strip():
            chunks.append(chunk)

        if start + chunk_words >= len(words):
            break

    return chunks


def index_pdf(pdf_bytes, filename):
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    documents = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        if len(pdf) == 0:
            raise ValueError("The PDF has no pages.")

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            page_text = page.get_text("text").strip()

            if not page_text:
                continue

            chunks = split_into_chunks(page_text)

            for chunk_index, chunk in enumerate(chunks):
                id_text = f"{pdf_hash}:{page_number}:{chunk_index}"
                document_id = hashlib.sha256(
                    id_text.encode("utf-8")
                ).hexdigest()

                documents.append(
                    Document(
                        id=document_id,
                        content=chunk,
                        meta={
                            "source": filename,
                            "page": page_number,
                            "chunk": chunk_index + 1,
                            "file_hash": pdf_hash,
                        },
                    )
                )

    if not documents:
        raise ValueError(
            "No selectable text was found. "
            "Scanned PDFs may need OCR before indexing."
        )

    embedded_result = document_embedder.run(documents=documents)
    embedded_documents = embedded_result["documents"]

    written_count = document_store.write_documents(
        embedded_documents,
        policy=DuplicatePolicy.OVERWRITE,
    )

    return len(documents), written_count


# ==================================================
# ANSWERS AND CITATION CHECKS
# ==================================================

def validate_citations(answer, sources):
    warnings = []

    valid_ids = {str(source["id"]) for source in sources}
    cited_ids = set(re.findall(r"\[(\d+)\]", answer))

    unknown_ids = sorted(cited_ids - valid_ids, key=int)

    if unknown_ids:
        warnings.append(
            "Citation numbers do not match retrieved sources: "
            + ", ".join(f"[{item}]" for item in unknown_ids)
        )

    if not cited_ids:
        warnings.append("No numbered source citations were detected.")

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

    uncited_sentences = [
        sentence
        for sentence in sentences
        if len(sentence) > 180
        and not re.search(r"\[\d+\]", sentence)
    ]

    if uncited_sentences:
        warnings.append(
            "Some long sentences lack numbered citations. "
            "Review them against the source passages."
        )

    if not warnings:
        warnings.append(
            "Citation format check passed. This does not verify "
            "that every claim is supported by its source."
        )

    return warnings


def get_answer(question):
    question = question.strip()

    if not question:
        return (
            "Please enter a question about monument conservation.",
            [],
            ["No question was provided."],
        )

    retrieval_result = retrieval_pipeline.run(
        {"query_embedder": {"text": question}}
    )

    retrieved_docs = retrieval_result["retriever"]["documents"]

    if not retrieved_docs:
        return (
            "I could not find relevant passages in the current "
            "knowledge base. Try rephrasing your question or "
            "uploading a relevant PDF.",
            [],
            ["No source passages were retrieved."],
        )

    sources = []

    for index, doc in enumerate(retrieved_docs[:TOP_K], start=1):
        metadata = doc.meta or {}

        sources.append({
            "id": index,
            "source": metadata.get("source", "Unknown source"),
            "page": metadata.get("page", "Unknown"),
            "text": doc.content or "",
            "score": doc.score,
        })

    source_context = "\n\n".join(
        f"[{source['id']}] Source: {source['source']}, "
        f"page {source['page']}\n{source['text']}"
        for source in sources
    )

    prompt = f"""
You are Heritage-RAG, an assistant for understanding
conservation and restoration guidance for historic monuments.

Answer using the retrieved source passages below.
Do not invent laws, official instructions, technical
specifications, or conservation procedures that are not
supported by the passages.

If the passages do not contain enough information,
say so clearly. For conservation decisions, recommend
consulting qualified conservation professionals and
the relevant official authority.

CITATION RULES:
- Cite supporting passages with their exact number, such as [1].
- Put citations next to the claims they support.
- Only use citation numbers listed in the passages.
- Do not invent sources, page numbers, or quotations.

RETRIEVED SOURCE PASSAGES:
{source_context}

USER QUESTION:
{question}

Write a clear, helpful answer with numbered citations:
"""

    try:
        generated = generator(
            prompt,
            max_new_tokens=350,
            do_sample=False,
            return_full_text=False,
        )

        answer = generated[0]["generated_text"].strip()

        if not answer:
            answer = (
                "The model did not generate an answer. "
                "Please try asking the question another way."
            )

    except Exception as exc:
        print("Answer-generation error:", repr(exc))
        answer = (
            "The answer-generation model encountered an error. "
            "Check the terminal for details."
        )

    warnings = validate_citations(answer, sources)

    return answer, sources, warnings


# ==================================================
# DOCUMENT MANAGEMENT HELPERS
# ==================================================

def get_managed_documents():
    all_docs = document_store.filter_documents()
    grouped = {}

    for doc in all_docs:
        meta = doc.meta or {}
        file_hash = meta.get("file_hash")

        # Do not manage legacy records without a file hash.
        if not file_hash:
            continue

        if file_hash not in grouped:
            grouped[file_hash] = {
                "file_hash": file_hash,
                "names": set(),
                "pages": set(),
                "chunk_count": 0,
            }

        item = grouped[file_hash]
        item["chunk_count"] += 1

        if meta.get("source"):
            item["names"].add(str(meta["source"]))

        if meta.get("page") is not None:
            item["pages"].add(meta["page"])

    results = []

    for item in grouped.values():
        names = sorted(item["names"])

        results.append({
            "file_hash": item["file_hash"],
            "filename": ", ".join(names) or "Unknown filename",
            "chunk_count": item["chunk_count"],
            "pages": sorted(item["pages"]),
        })

    return sorted(results, key=lambda item: item["filename"].lower())


def get_chunks_for_file(file_hash):
    all_docs = document_store.filter_documents()

    return [
        doc
        for doc in all_docs
        if (doc.meta or {}).get("file_hash") == file_hash
    ]


# ==================================================
# WEBSITE STYLING
# ==================================================

CSS = """
:root {
    --green: #173d32;
    --green-dark: #102b24;
    --cream: #f5f0e4;
    --white: #fffdf7;
    --gold: #b99454;
    --text: #24372f;
    --muted: #64746b;
    --border: #d8dfd4;
    --danger: #a83b32;
    --success: #286746;
}

* {
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    margin: 0;
    background-color: #e9e0d0;
    background-image:
        linear-gradient(
            rgba(247, 244, 236, 0.24),
            rgba(247, 244, 236, 0.42)
        ),
        url("https://images.unsplash.com/photo-1564507592333-c60657eea523?auto=format&fit=crop&w=2000&q=85");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    color: var(--text);
    font-family: Inter, Arial, sans-serif;
    line-height: 1.6;
}

main {
    width: min(1100px, 94%);
    margin: 0 auto;
    padding: 28px 0 48px;
}

h1, h2, h3 {
    color: var(--green);
    line-height: 1.2;
}

h1 {
    font-size: clamp(2rem, 5vw, 3.3rem);
}

h2 {
    font-size: clamp(1.4rem, 3vw, 2rem);
}

p {
    margin: 0.55rem 0;
}

a {
    color: var(--green);
    font-weight: 600;
}

.hero {
    padding: clamp(28px, 6vw, 56px);
    border-radius: 22px;
    background: rgba(23, 55, 44, 0.94);
    color: var(--white);
    border-bottom: 4px solid var(--gold);
    box-shadow: 0 10px 30px rgba(20, 35, 26, 0.18);
}

.hero h1,
.hero h2,
.hero h3 {
    color: var(--white);
}

.hero p {
    color: #f0eadb;
    max-width: 760px;
}

.panel {
    margin: 22px 0;
    padding: clamp(18px, 4vw, 30px);
    border: 1px solid rgba(255, 255, 255, 0.75);
    border-radius: 18px;
    background: rgba(255, 253, 247, 0.92);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    box-shadow: 0 8px 28px rgba(35, 49, 38, 0.14);
}

input[type="text"],
input[type="file"],
textarea,
select {
    width: 100%;
    max-width: 100%;
    padding: 12px 14px;
    margin: 6px 0 12px;
    border: 1px solid #cbd5ca;
    border-radius: 9px;
    background: #fffefa;
    color: var(--text);
    font: inherit;
}

textarea {
    min-height: 110px;
    resize: vertical;
}

input:focus,
textarea:focus,
select:focus {
    outline: 3px solid rgba(185, 148, 84, 0.28);
    border-color: var(--gold);
}

button {
    padding: 12px 18px;
    border: 0;
    border-radius: 9px;
    background: var(--green);
    color: white;
    font: inherit;
    font-weight: 700;
    cursor: pointer;
    transition: background 0.2s ease, transform 0.2s ease;
}

button:hover {
    background: var(--green-dark);
    transform: translateY(-1px);
}

button:focus-visible,
a:focus-visible {
    outline: 3px solid var(--gold);
    outline-offset: 3px;
}

hr {
    border: 0;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    background: rgba(255, 255, 255, 0.65);
    border-radius: 10px;
    overflow: hidden;
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}

th {
    background: #e6ecdf;
    color: var(--green);
}

.success {
    padding: 18px;
    border: 1px solid #b8d9c2;
    border-radius: 12px;
    background: #edf7ee;
    color: #205a37;
}

.danger {
    padding: 18px;
    border: 1px solid #e3c0bc;
    border-radius: 12px;
    background: #fff0ed;
    color: #842e27;
}

.source-card {
    margin: 14px 0;
    padding: 16px;
    border: 1px solid var(--border);
    border-left: 4px solid var(--gold);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.78);
}

.source-meta {
    color: var(--muted);
    font-size: 0.92rem;
    margin-bottom: 8px;
}

.warning {
    margin: 10px 0;
    padding: 12px 14px;
    border-left: 4px solid var(--gold);
    background: #fff7df;
    border-radius: 7px;
}

footer {
    padding: 22px 12px;
    color: #f7f1e5;
    text-align: center;
    background: rgba(16, 43, 36, 0.95);
}

footer a {
    color: #f3dba8;
}

@media (max-width: 650px) {
    main {
        width: 94%;
        padding-top: 16px;
    }

    .hero, .panel {
        padding: 18px;
        border-radius: 14px;
    }

    table {
        display: block;
        overflow-x: auto;
        white-space: nowrap;
    }

    body {
        background-attachment: scroll;
    }
}
"""


# ==================================================
# FASTHTML APP SETUP
# ==================================================

app, rt = fast_app(
    hdrs=(
        Link(
            rel="preconnect",
            href="https://fonts.googleapis.com",
        ),
        Link(
            rel="preconnect",
            href="https://fonts.gstatic.com",
            crossorigin="anonymous",
        ),
        Link(
            rel="stylesheet",
            href=(
                "https://fonts.googleapis.com/css2?"
                "family=Inter:wght@400;500;600;700&display=swap"
            ),
        ),
    )
)


# ==================================================
# HOME PAGE
# ==================================================

@rt("/")
def home():
    total_chunks = document_store.count_documents()

    return Titled(
        "Heritage-RAG — Historic Conservation Assistant",
        Style(CSS),
        Main(
            Div(
                H1("Heritage-RAG"),
                P(
                    "A source-backed assistant for historic monument "
                    "conservation and restoration research."
                ),
                P(
                    "Explore information from your indexed documents. "
                    "Verify restoration decisions with qualified "
                    "professionals and official guidance."
                ),
                cls="hero",
            ),
            Div(
                H2("Ask a question"),
                Form(
                    Input(
                        type="text",
                        name="question",
                        placeholder="Ask about conservation guidance...",
                        required=True,
                    ),
                    Button("Ask Heritage-RAG", type="submit"),
                    method="post",
                    action="/ask",
                ),
                H3("Example questions"),
                Ul(
                    Li("What materials are mentioned for historic masonry conservation?"),
                    Li("What does the source say about lime mortar?"),
                    Li("What precautions are described for monument restoration?"),
                    Li("Which conservation guidelines are discussed in the uploaded PDFs?"),
                ),
                cls="panel",
            ),
            Div(
                H2("Upload a PDF"),
                P("Upload a text-based conservation PDF to index it."),
                P("Maximum file size: 15 MB. Scanned documents may require OCR first."),
                Form(
                    Input(
                        type="file",
                        name="pdf_file",
                        accept=".pdf,application/pdf",
                        required=True,
                    ),
                    Button("Upload and index PDF", type="submit"),
                    method="post",
                    action="/upload",
                    enctype="multipart/form-data",
                ),
                P(f"Currently indexed chunks: {total_chunks}"),
                cls="panel",
            ),
            Div(
                H2("Quick Features"),
                Ul(
                    Li("Ask questions about conservation and restoration"),
                    Li("Get answers with source citations and page numbers"),
                    Li("Upload and index your own PDF documents"),
                    Li(A("Manage and delete indexed documents", href="/manage")),
                ),
                cls="panel",
            ),
            Footer(
                P(
                    "Heritage-RAG is a research aid, not a replacement "
                    "for professional conservation advice."
                )
            ),
        ),
    )


# ==================================================
# QUESTION AND ANSWER PAGE
# ==================================================

@rt("/ask", methods=["POST"])
def ask(question: str):
    answer, sources, warnings = get_answer(question)

    source_cards = []

    for source in sources:
        source_cards.append(
            Div(
                H3(f"Source [{source['id']}]"),
                P(
                    Strong("Document: "),
                    source["source"],
                ),
                P(
                    Strong("Page: "),
                    str(source["page"]),
                ),
                P(source["text"]),
                cls="source-card",
            )
        )

    warning_cards = [
        Div(P(warning), cls="warning")
        for warning in warnings
    ]

    return Titled(
        "Answer — Heritage-RAG",
        Style(CSS),
        Main(
            Div(
                H1("Your question"),
                P(question),
                H2("Answer"),
                P(answer),
                cls="panel",
            ),
            Div(
                H2("Retrieved source passages"),
                *(
                    source_cards
                    if source_cards
                    else [P("No source passages were retrieved.")]
                ),
                cls="panel",
            ),
            Div(
                H2("Citation checks"),
                *warning_cards,
                P(
                    "Citation-format checks are not a guarantee that "
                    "the model's claims are fully supported."
                ),
                cls="panel",
            ),
            A("Back to Home", href="/"),
        ),
    )


# ==================================================
# PDF UPLOAD ROUTE
# ==================================================

@rt("/upload", methods=["POST"])
async def upload_pdf(pdf_file: UploadFile):
    filename = Path(
        pdf_file.filename or "uploaded.pdf"
    ).name

    try:
        if Path(filename).suffix.lower() != ".pdf":
            raise ValueError("Please upload a PDF file.")

        pdf_bytes = await pdf_file.read(MAX_PDF_SIZE + 1)

        if len(pdf_bytes) > MAX_PDF_SIZE:
            raise ValueError("The file exceeds the 15 MB size limit.")

        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError(
                "The uploaded file does not appear to be a PDF."
            )

        chunks_created, chunks_written = index_pdf(
            pdf_bytes,
            filename,
        )

        return Div(
            P(f"PDF indexed: {filename}"),
            P(f"Chunks processed: {chunks_created}"),
            P(f"Chunks written: {chunks_written}"),
            P(
                "Total indexed chunks: "
                f"{document_store.count_documents()}"
            ),
            P(A("Open Document Management", href="/manage")),
            cls="success",
        )

    except Exception as exc:
        print("PDF upload error:", repr(exc))

        return Div(
            P(f"Could not index the PDF: {exc}"),
            cls="danger",
        )

    finally:
        await pdf_file.close()


# ==================================================
# DOCUMENT MANAGEMENT PAGE
# ==================================================

@rt("/manage")
def manage_documents():
    all_docs = document_store.filter_documents()
    files = get_managed_documents()

    legacy_count = sum(
        1
        for doc in all_docs
        if not (doc.meta or {}).get("file_hash")
    )

    if files:
        document_table = Table(
            Thead(
                Tr(
                    Th("PDF filename"),
                    Th("Chunks"),
                    Th("Pages represented"),
                )
            ),
            Tbody(
                *[
                    Tr(
                        Td(item["filename"]),
                        Td(str(item["chunk_count"])),
                        Td(
                            ", ".join(map(str, item["pages"])) or "Unknown"
                        ),
                    )
                    for item in files
                ]
            ),
        )

        options = [
            Option(
                f"{item['filename']} — {item['chunk_count']} chunks",
                value=item["file_hash"],
            )
            for item in files
        ]

        delete_form = Form(
            H3("Delete an indexed PDF"),
            P("Select a document to review its deletion details."),
            Select(*options, name="file_hash", required=True),
            Button("Review selected PDF", type="submit"),
            method="post",
            action="/delete-preview",
        )

    else:
        document_table = P("No manageable indexed PDFs were found.")
        delete_form = P(
            "There are no PDFs available to select for deletion."
        )

    return Titled(
        "Document Management — Heritage-RAG",
        Style(CSS),
        Main(
            Div(
                H1("Document Management"),
                P(f"Managed PDFs: {len(files)}"),
                P(
                    "Total indexed chunks: "
                    f"{document_store.count_documents()}"
                ),
                document_table,
                Hr(),
                delete_form,
                Hr(),
                P(
                    f"Legacy chunks without file_hash: {legacy_count}. "
                    "These records cannot be deleted from this page."
                ),
                P(
                    "This page manages indexed chunks. The original "
                    "uploaded PDF files are not stored by this app."
                ),
                A("Back to Heritage-RAG", href="/"),
                cls="panel",
            )
        ),
    )


# ==================================================
# DELETION PREVIEW AND CONFIRMATION
# ==================================================

@rt("/delete-preview", methods=["POST"])
def delete_preview(file_hash: str):
    chunks = get_chunks_for_file(file_hash)

    if not chunks:
        return Titled(
            "Document not found",
            P("No matching chunks were found. Nothing was deleted."),
            A("Return to management", href="/manage"),
        )

    if any(
        (doc.meta or {}).get("file_hash") != file_hash
        for doc in chunks
    ):
        return Titled(
            "Safety check failed",
            P("Nothing was deleted."),
            A("Return to management", href="/manage"),
        )

    names = sorted({
        str((doc.meta or {}).get("source"))
        for doc in chunks
        if (doc.meta or {}).get("source")
    })

    expected_name = ", ".join(names) or "Unknown filename"

    return Titled(
        "Confirm PDF deletion",
        Style(CSS),
        Main(
            Div(
                H1("Confirm PDF deletion"),
                P("Review these details carefully."),
                P(Strong("PDF: "), expected_name),
                P(Strong("Chunks to delete: "), str(len(chunks))),
                Div(
                    P(
                        "Only chunks matching this PDF's exact content "
                        "hash will be selected. This action is permanent."
                    ),
                    cls="danger",
                ),
                Form(
                    Input(
                        type="hidden",
                        name="file_hash",
                        value=file_hash,
                    ),
                    Label(
                        "Type this exact filename to confirm: "
                        + expected_name
                    ),
                    Input(
                        type="text",
                        name="confirm_name",
                        placeholder="Type the exact filename",
                        required=True,
                    ),
                    Button(
                        "Confirm permanent deletion",
                        type="submit",
                    ),
                    method="post",
                    action="/delete",
                ),
                P(A("Cancel", href="/manage")),
                cls="panel",
            )
        ),
    )


# ==================================================
# DELETE SELECTED DOCUMENT CHUNKS
# ==================================================

@rt("/delete", methods=["POST"])
def delete_document(file_hash: str, confirm_name: str):
    chunks = get_chunks_for_file(file_hash)

    if not chunks:
        return Titled(
            "Nothing deleted",
            P("No matching chunks were found."),
            A("Return to management", href="/manage"),
        )

    names = sorted({
        str((doc.meta or {}).get("source"))
        for doc in chunks
        if (doc.meta or {}).get("source")
    })

    expected_name = ", ".join(names) or "Unknown filename"

    if confirm_name.strip() != expected_name:
        return Titled(
            "Confirmation did not match",
            P("The filename did not match. Nothing was deleted."),
            P(f"Expected filename: {expected_name}"),
            A("Return to management", href="/manage"),
        )

    # Verify every chunk before deleting anything.
    if any(
        (doc.meta or {}).get("file_hash") != file_hash
        for doc in chunks
    ):
        return Titled(
            "Safety check failed",
            P("Nothing was deleted because validation failed."),
            A("Return to management", href="/manage"),
        )

    ids_to_delete = [doc.id for doc in chunks]

    if not ids_to_delete:
        return Titled(
            "Nothing deleted",
            P("No chunk IDs were found."),
            A("Return to management", href="/manage"),
        )

    # Delete only the IDs belonging to this exact hash.
    document_store.delete_documents(ids_to_delete)

    # Verify that no chunks with this hash remain.
    remaining = get_chunks_for_file(file_hash)

    if remaining:
        return Titled(
            "Deletion needs checking",
            P(
                "Some matching chunks remain. Check the terminal "
                "and management page."
            ),
            A("Return to management", href="/manage"),
        )

    return Titled(
        "Document deleted",
        Style(CSS),
        Main(
            Div(
                H1("Deletion complete"),
                P(f"Selected PDF: {expected_name}"),
                P(f"Chunks removed: {len(ids_to_delete)}"),
                P(
                    "Chunks remaining in knowledge base: "
                    f"{document_store.count_documents()}"
                ),
                A("Back to Document Management", href="/manage"),
                cls="success",
            )
        ),
    )


# ==================================================
# START THE APPLICATION
# ==================================================

if __name__ == "__main__":
    # Use port 5001 locally; respect PORT if provided by a host.
    serve(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5001")),
        reload=False,
    )