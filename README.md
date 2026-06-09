# ⚗️ ChemE RAG Study Assistant

A Retrieval-Augmented Generation (RAG) chatbot for Chemical Engineering students at IIT Roorkee. Upload your lecture PDFs and ask questions — the assistant answers strictly from your materials.

## What It Does

1. **Upload** Chemical Engineering PDFs (NPTEL notes, textbooks, lecture slides)
2. **Process** — text is extracted, chunked, and embedded into a FAISS vector store
3. **Ask** — type a question, and the RAG pipeline retrieves relevant chunks + generates an answer using Google Gemini

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini 1.5 Flash (free) |
| Embeddings | Google `embedding-001` |
| Vector Store | FAISS (CPU) |
| Framework | LangChain |
| PDF Parsing | PyPDF2 |

## Key Modifications from Original

| Original (alejandro-ao/ask-multiple-pdfs) | This Version |
|---|---|
| OpenAI GPT (paid) | Google Gemini 1.5 Flash (free) |
| OpenAI Embeddings (paid) | Google `embedding-001` (free) |
| Generic chatbot | ChemE domain with system prompt |
| No domain data | NPTEL Chemical Engineering PDFs |
| No input validation | Warns if no PDF uploaded, skips blank pages |
| Generic page title | "ChemE Study Assistant — IIT Roorkee" |
| No chunk stats | Shows chunks indexed after processing |

## Setup

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/cheme-rag-assistant
cd cheme-rag-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get your free API key from https://aistudio.google.com/app/apikey
#    Then add it to .env:
echo "GOOGLE_API_KEY=your_key_here" > .env

# 4. Run the app
streamlit run app.py
```

## Architecture (RAG Pipeline)

```
PDF Upload → Text Extraction (PyPDF2)
           → Chunking (CharacterTextSplitter, 1000 chars, 200 overlap)
           → Embedding (Google embedding-001)
           → Vector Store (FAISS)

User Question → Retrieve top-4 chunks (FAISS similarity search)
              → Augment prompt with chunks + system context
              → Generate answer (Gemini 1.5 Flash)
              → Display with conversation memory
```

## Project Structure

```
cheme-rag-assistant/
├── app.py              ← Main Streamlit application
├── htmlTemplates.py    ← Chat UI templates
├── requirements.txt    ← Python dependencies
├── .env                ← API key (not committed)
├── .gitignore
└── README.md
```

## License

MIT
