import streamlit as st
import PyPDF2
import google.generativeai as genai
import psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

NEON_DB_URL = os.getenv("NEON_CONNECTION_STRING")
conn = psycopg2.connect(NEON_DB_URL)
cursor = conn.cursor()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

# PDF Extraction
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text() is not None])

# Store embeddings in DB
def store_embeddings(source, text):
    cursor.execute("SELECT 1 FROM embeddings WHERE source = %s LIMIT 1", (source,))
    if cursor.fetchone():
        return
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    for i, chunk in enumerate(chunks):
        embedding = embed_model.encode(chunk).tolist()
        cursor.execute("INSERT INTO embeddings (source, chunk_id, content, embedding) VALUES (%s, %s, %s, %s)",
                       (source, i, chunk, Json(embedding)))
    conn.commit()

# Search embeddings
def search_embeddings(query, top_k=10):
    query_embedding = embed_model.encode(query).tolist()
    cursor.execute("SELECT source, content FROM embeddings ORDER BY embedding <-> %s LIMIT %s;", (Json(query_embedding), top_k))
    return cursor.fetchall()

# Ask Gemini AI
def ask_gemini(question, context):
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat_history = "\n".join(st.session_state.chat_history[-5:])
    prompt = f"""
    Chat History:
    {chat_history}
    
    Context:
    {context}
    
    Question: {question}

    """
    response = model.generate_content(prompt)
    answer = response.text.strip() if response.text else "No response from Gemini."
    st.session_state.chat_history.append(f"Q: {question}\nA: {answer}")
    return answer

# Page Configuration
st.set_page_config(page_title="AI Chatbot", layout="wide")

# Sidebar for Chat History
st.sidebar.markdown("## 🗂️ Chat History")
for chat in st.session_state.chat_history[-10:]:
    st.sidebar.markdown(f"**{'👤 You:' if chat.startswith('Q:') else '🤖 AI:'}** {chat[3:]}")

# Title
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>💬 AI Chatbot</h1>", unsafe_allow_html=True)

# Upload Section - Side by Side
col1, col2 = st.columns([2,1])

with col1:
    uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        st.session_state.extracted_text = extract_text_from_pdf(uploaded_file)
        store_embeddings(uploaded_file.name, st.session_state.extracted_text)
        st.success("✅ PDF processed successfully!")

with col2:
    website_url = st.text_input("🌐 Enter Website URL")
    if website_url:
        st.success(f"✅ Website URL saved: {website_url}")

st.markdown("---")  # Simple horizontal line

# Chat Messages Display
for chat in st.session_state.chat_history[-10:]:
    color = "#E8F0FE" if chat.startswith("Q:") else "#F0F0F0"
    st.markdown(f"""
    <div style='background-color: {color}; padding: 10px; border-radius: 10px; margin: 5px 0;'>
        <strong>{'👤 You:' if chat.startswith('Q:') else '🤖 AI:'}</strong> {chat[3:]}
    </div>
    """, unsafe_allow_html=True)

# Fixed Input Box at Bottom
query = st.chat_input("💬 Type your message here...")
if query:
    with st.spinner("Thinking..."):
        top_results = search_embeddings(query, top_k=10)
        sources = "\n".join([f"Source: {src}\nContent: {txt}..." for src, txt in top_results]) if top_results else ""
        answer = ask_gemini(query, sources)
        st.session_state.chat_history.append(f"Q: {query}\nA: {answer}")
        st.write("**🤖 AI:**", answer)
