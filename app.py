import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template
import os
import time

# Path to save/load the FAISS index
FAISS_INDEX_DIR = "faiss_index"


SYSTEM_CONTEXT = """You are an academic assistant for students.
You answer questions strictly based on the uploaded documents.
Always mention which concept or topic your answer is based on.
If the answer is not in the documents, say: 'This is not covered in the uploaded materials.'
"""


def get_pdf_text(pdf_docs):
    """Extract raw text from all uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted: 
                text += extracted
    return text


def get_text_chunks(text):
    """Split text into overlapping chunks for retrieval."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,      # larger chunks = fewer API calls (important for free tier)
        chunk_overlap=300,    # overlap so answers don't get cut off at boundaries
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """Convert chunks to embeddings and store in FAISS.
    Batches requests with retry logic to handle free tier rate limits.
    """
    # YOUR CHANGE 3: Using Google embeddings (free) instead of OpenAI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Small batches + long delay to stay under 100 req/min free tier limit
    batch_size = 20
    vectorstore = None
    total_batches = (len(text_chunks) + batch_size - 1) // batch_size

    progress_bar = st.progress(0, text="Embedding chunks...")

    for batch_num, i in enumerate(range(0, len(text_chunks), batch_size)):
        batch = text_chunks[i:i + batch_size]
        progress = (batch_num + 1) / total_batches
        progress_bar.progress(progress, text=f"Embedding batch {batch_num + 1}/{total_batches}...")

        # Retry up to 5 times with exponential backoff on rate limit
        for attempt in range(5):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_texts(texts=batch, embedding=embeddings)
                else:
                    vectorstore.add_texts(batch)
                break  # success, move to next batch
            except Exception as e:
                err_str = str(e)
                if any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED", "500", "INTERNAL"]):
                    wait_time = 20 * (attempt + 1)
                    progress_bar.progress(progress, text=f"Rate limited - waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        # Pause between batches to avoid hitting the limit
        if i + batch_size < len(text_chunks):
            time.sleep(15)

    progress_bar.empty()

    
    vectorstore.save_local(FAISS_INDEX_DIR)

    return vectorstore


def load_vectorstore():
    """Load a previously saved FAISS index from disk."""
    if os.path.exists(FAISS_INDEX_DIR):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        return FAISS.load_local(
            FAISS_INDEX_DIR, embeddings,
            allow_dangerous_deserialization=True
        )
    return None


def ask_question(vectorstore, question, chat_history):
    """Answer a question using RAG: retrieve docs + call LLM once.
    This uses only 1 LLM call + 1 embedding call per question
    (vs 2 LLM calls + 1 embedding in ConversationalRetrievalChain).
    """
    # Step 1: Retrieve relevant chunks (1 embedding API call)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    for attempt in range(5):
        try:
            docs = retriever.invoke(question)
            break
        except Exception as e:
            err_str = str(e)
            if any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "503"]):
                wait_time = 10 * (attempt + 1)
                st.warning(f"Retrieval API error (attempt {attempt + 1}/5) - waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    else:
        return None

    # Step 2: Build context from retrieved docs
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # Step 3: Build the prompt with system context + retrieved docs + question
    # Include last 3 Q&A pairs for conversational context (no extra LLM call needed)
    history_text = ""
    if chat_history:
        recent = chat_history[-6:]  # last 3 Q&A pairs
        for i in range(0, len(recent), 2):
            if i + 1 < len(recent):
                history_text += f"Previous Q: {recent[i]}\nPrevious A: {recent[i+1]}\n\n"

    full_prompt = f"""{SYSTEM_CONTEXT}

Context from uploaded documents:
{context}

{f"Recent conversation:{chr(10)}{history_text}" if history_text else ""}
Student's question: {question}

Provide a helpful, accurate answer based on the context above."""

    # Step 4: Call LLM once (1 API call)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )

    for attempt in range(5):
        try:
            response = llm.invoke(full_prompt)
            return response.content
        except Exception as e:
            err_str = str(e)
            if any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "503"]):
                wait_time = 10 * (attempt + 1)
                st.warning(f"LLM API error (attempt {attempt + 1}/5) - waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                st.error(f"Error: {e}")
                return None

    return None


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Study Assistant",  
        page_icon=":alembic:"
    )
    st.write(css, unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

  
    if st.session_state.vectorstore is None:
        saved_vs = load_vectorstore()
        if saved_vs is not None:
            st.session_state.vectorstore = saved_vs
            st.sidebar.success("Loaded saved knowledge base from disk.")

   
    st.header("RAG Study Assistant")
    st.markdown("*Ask questions across your PDFs*")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.vectorstore is None:
            st.warning("Please upload and process PDFs first.")
        else:
            # Show previous chat history
            for i in range(0, len(st.session_state.chat_history), 2):
                st.write(user_template.replace("{{MSG}}", st.session_state.chat_history[i]),
                         unsafe_allow_html=True)
                if i + 1 < len(st.session_state.chat_history):
                    st.write(bot_template.replace("{{MSG}}", st.session_state.chat_history[i + 1]),
                             unsafe_allow_html=True)

            # Ask the new question
            answer = ask_question(
                st.session_state.vectorstore,
                user_question,
                st.session_state.chat_history
            )

            if answer:
                # Show current Q&A
                st.write(user_template.replace("{{MSG}}", user_question),
                         unsafe_allow_html=True)
                st.write(bot_template.replace("{{MSG}}", answer),
                         unsafe_allow_html=True)

                # Save to history
                st.session_state.chat_history.append(user_question)
                st.session_state.chat_history.append(answer)
            else:
                st.error("Could not get an answer. Please wait a minute and try again.")

    with st.sidebar:
        st.subheader("Upload Your Study Material")
        pdf_docs = st.file_uploader(
            "Upload PDF notes (lecture_notes, textbooks, etc.)",
            accept_multiple_files=True,
            type=['pdf']
        )
        if st.button("Process Documents"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Building knowledge base..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chat_history = []
                    
                    st.success(f"Processed {len(pdf_docs)} PDF(s) - {len(text_chunks)} chunks indexed")

        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == '__main__':
    main()
