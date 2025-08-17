import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

# Load Q&A model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("📄 Dynamic Smart Q&A Chatbot")
st.write("Upload a document and ask questions about it!")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    # Read file content
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    else:
        text = uploaded_file.read().decode("utf-8")

    # Chunk text
    words = text.split()
    chunk_size = 500
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    # Build FAISS index
    embeddings = embed_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Question input
    question = st.text_input("Enter your question:")

    if question:
        # Search top chunks
        query_vec = embed_model.encode([question])
        D, I = index.search(query_vec, 3)
        top_chunks = [chunks[i] for i in I[0]]

        # Combine chunks and get answer
        combined_context = " ".join(top_chunks)
        result = qa_pipeline(question=question, context=combined_context)

        st.subheader("Answer:")
        st.write(result['answer'])
        st.caption(f"Confidence: {result['score']:.2f}")