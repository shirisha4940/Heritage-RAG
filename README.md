# Heritage-RAG
### AI-Powered Knowledge Assistant for Historic Monument Conservation

Heritage-RAG is a Retrieval-Augmented Generation (RAG) application designed to help users explore information from documents related to historic monuments and cultural heritage conservation.

The application combines document retrieval with a language model to generate context-aware answers grounded in indexed documents, helping users find relevant information more efficiently.

## Features

- **PDF Upload and Indexing:** Upload PDF documents for processing and retrieval.
- **Semantic Search:** Find relevant passages using text embeddings.
- **AI-Powered Question Answering:** Generate answers using retrieved document context.
- **Source Citations:** Display available source references to help verify answers.
- **Document Management:** Manage indexed documents and preview deletion actions.
- **Persistent Vector Storage:** Store document embeddings locally using Chroma.
- **Web Interface:** Interact with the application through a Python-based interface.

## Technology Stack

- Python
- FastHTML
- Haystack
- ChromaDB
- Sentence Transformers
- Qwen2.5-0.5B-Instruct
- PDF document processing

## How It Works

1. Upload a PDF document containing information about historic monuments or conservation.
2. Extract and process the document text.
3. Generate vector embeddings for document passages.
4. Store the embeddings in ChromaDB.
5. Ask a question through the web interface.
6. Retrieve relevant passages to provide context to the language model.
7. Display the generated answer with available source references.

## Getting Started

### Prerequisites

- Python installed on your computer
- Git installed on your computer
- Internet access to download dependencies and language models, if needed

### 1. Clone the Repository

```bash
git clone https://github.com/shirisha4940/Heritage-RAG.git
cd Heritage-RAG
```

### 2. Create a Virtual Environment

For Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```cmd
pip install -r requirements.txt
```

### 4. Run the Application

If `app.py` is the application entry point, run:

```cmd
python app.py
```

Open the local URL displayed by the application in your browser. The current development configuration may use:

`http://127.0.0.1:5001`

## Example Use Cases

- Finding relevant information in historic monument documentation.
- Exploring conservation and restoration reports.
- Locating supporting passages in heritage research documents.
- Supporting research into cultural heritage preservation.

## Project Structure

```text
Heritage-RAG/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── Dockerfile
└── ...
```

The repository may contain additional Python modules depending on the project configuration. Local documents, vector databases, and secret files should remain excluded from the public repository.

## Privacy and Security

- Do not upload private documents, passwords, session keys, API keys, or other secrets.
- Keep local document collections and vector databases out of public version control.
- Verify generated answers against their cited sources.
- Configure authentication and appropriate security protections before exposing the application to other users.

## Future Improvements

- Improve citation validation and answer evaluation.
- Add support for more document formats.
- Add automated tests.
- Improve retrieval accuracy.
- Add authentication and access controls for multi-user deployments.
- Prepare secure deployment documentation.

## Disclaimer

Heritage-RAG is an experimental learning and research project intended to support exploration of cultural heritage documentation. It does not replace professional conservation advice or expert assessment.

## Author

**Shirisha**

GitHub: https://github.com/shirisha4940

---

*Exploring how Retrieval-Augmented Generation can improve access to historic monument and conservation knowledge.*
