# 📄 Smart Q&A Chatbot with Transformers

A **dynamic question-answering chatbot** that allows users to upload their own documents (PDF or TXT) and get answers based on the content. Built using **Hugging Face Transformers**, **Sentence-Transformers**, **FAISS**, and **Streamlit**, this chatbot demonstrates practical applications of AI-powered document understanding.

---

## **Features**

- ✅ Upload any PDF or TXT document for context
- ✅ Semantic search using **embeddings** to find relevant chunks
- ✅ Q&A powered by a **state-of-the-art transformer model** (`deepset/roberta-base-squad2`)
- ✅ Real-time answers with confidence scores
- ✅ Easy-to-use **web interface** via Streamlit
- ✅ Fully **dynamic** – works with any uploaded document

---

## **Demo**

Run locally:

```bash
streamlit run app.py