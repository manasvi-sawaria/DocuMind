# DocuMind - Retrival - Augmented Document Intelligence System

> Ask questions across any set of PDFs — grounded answers powered by Retrieval-Augmented Generation, Google Gemini 2.5 Flash, and FAISS vector search.

---

## Features

- **Upload Multiple PDFs:** Upload any number of textbooks, notes, or slides simultaneously.
- **Persistent Knowledge Base:** The processed index is automatically saved locally to a `faiss_index/` folder and auto-loaded on app startup so you don't need to re-upload PDFs every time you reload the page.
- **Robust Error Handling:** Embedded retry logic with exponential backoff handles rate-limiting (`429`) and transient server (`500`/`503`/`INTERNAL`) errors gracefully.
- **API Call Optimization:** Custom prompt injection keeps chat memory without calling extra LLM rephrasing chains, reducing Gemini API calls by 33%.
- **Strictly Grounded Answers:** Tailored system context forces the assistant to answer strictly using document content, citing the topic and refusing outside materials.
- **UI Customizations:** Added stats tracker on upload, custom HTML chat templates, and a "Clear Chat History" sidebar option.

---

## How It Works

```
                     INDEXING PHASE                      
                                                         
  PDF Upload ──► Text Extraction ──► Chunking            
                   (PyPDF2)        (1000 chars,          
                                    200 overlap)         
                       │                                 
                       ▼                                 
               Google Embeddings ──► FAISS Vector Store  
             (gemini-embedding-001)                      
                       │                                 
                       ▼                                 
               Saved locally to disk (faiss_index/)     


                     QUERY PHASE                         
                                                         
  User Question ──► Embed Query ──► FAISS Search         
                                   (top-4 chunks)        
                                        │                
                                        ▼                
                             Augment Prompt             
                        (chunks + custom history        
                           + system context)             
                                        │                
                                        ▼                
                          Gemini 2.5 Flash ──► Answer   

```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | Streamlit | Rapid interactive UI |
| LLM | Google Gemini 2.5 Flash | Fast, high-quality reasoning |
| Embeddings | Google `gemini-embedding-001` | Strong semantic representation |
| Vector Store | FAISS (CPU) | Efficient similarity search; local storage/caching |
| RAG Framework | LangChain | Core modular components |
| PDF Parsing | PyPDF2 | Lightweight text extraction |

---

## Quickstart

### 1. Clone and enter the repo
```bash
git clone https://github.com/manasvi-sawaria/rag-document-intellegence.git
cd rag-document-intellegence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a file named `.env` in the root directory and add your Google API key:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(Get a free key from the [Google AI Studio Console](https://aistudio.google.com/app/apikey).)*

### 4. Run the app
```bash
streamlit run app.py
```

---

## Project Structure

```
rag-document-intellegence/
│
├── app.py                # Main Streamlit app & RAG pipeline
├── htmlTemplates.py      # Custom CSS & message components
├── requirements.txt      # Python dependencies
├── faiss_index/          # Local FAISS database (created after processing)
├── .env                  # Environment keys (ignored by git)
├── .gitignore            # Excludes keys and cache from commits
└── README.md
```

---


