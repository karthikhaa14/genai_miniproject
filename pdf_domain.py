import streamlit as st
import PyPDF2
import google.generativeai as genai
import psycopg2
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# NeonDB Connection
NEON_DB_URL = os.getenv("NEON_CONNECTION_STRING")
conn = psycopg2.connect(NEON_DB_URL)
cursor = conn.cursor()

# Initialize SentenceTransformer model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

# Function to scrape website content
def scrape_website(url):
    driver = webdriver.Edge()
    driver.get(url)
    time.sleep(3)
    try:
        page_content = driver.find_element(By.TAG_NAME, "body").text
    except:
        page_content = "Could not extract data."
    driver.quit()
    return page_content

# Function to store embeddings
def store_embeddings(source, text):

    # Check if any embeddings exist for this source
    cursor.execute("SELECT 1 FROM embeddings WHERE source = %s LIMIT 1", (source,))
    exists = cursor.fetchone()  # Will be None if no record exists

    if exists:
        print(f"Embeddings for {source} already exist. Skipping storage.")
        return  # Skip storage if embeddings already exist

    # If embeddings do not exist, proceed with storing them
    chunks = split_text_into_chunks(text)
    for i, chunk in enumerate(chunks):
        embedding = embed_model.encode(chunk).tolist()
        cursor.execute(
            "INSERT INTO embeddings (source, chunk_id, content, embedding) VALUES (%s, %s, %s, %s)",
            (source, i, chunk, Json(embedding))
        )
    conn.commit()
    print(f"Embeddings stored for {source}.")


# Function to retrieve relevant chunks
def search_embeddings(query, top_k=10):
    query_embedding = embed_model.encode(query).tolist()
    cursor.execute("""
    SELECT source, content FROM embeddings
    ORDER BY embedding <-> %s
    LIMIT %s;
    """, (Json(query_embedding), top_k))
    return cursor.fetchall()

# Function to query Gemini API
def ask_gemini(question, context):
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat_history = "\n".join(st.session_state.chat_history[-5:])
    prompt = f"""
    Chat History:
    {chat_history}

    Context:
    {context}

    Question: {question}
    Answer: If you can't find the answer, say that I couldn't find the answer.
    """
    response = model.generate_content(prompt)
    answer = response.text.strip() if response.text else "No response from Gemini."
    st.session_state.chat_history.append(f"Q: {question}\nA: {answer}")
    return answer

# Streamlit Layout
st.set_page_config(page_title="AI Chatbot", layout="wide")

with st.sidebar:
    st.title("Chat History")
    for chat in st.session_state.chat_history[-10:]:
        st.write(chat)

st.title("AI-Powered Search on PDFs & Websites")

col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:  # Loop through multiple files
            pdf_text = extract_text_from_pdf(uploaded_file)  # Extract text for each file
            store_embeddings(uploaded_file.name, pdf_text)  # Store embeddings
        st.success("PDF content chunked, embedded, and stored.")

with col2:
    website_url = st.text_input("Enter Website URL")
    if st.button("Scrape Website"):
        site_content = scrape_website(website_url)
        store_embeddings(website_url, site_content)
        st.success("Website content chunked, embedded, and stored.")

query = st.text_input("Ask a question")
if st.button("Search & Answer"):
    top_results = search_embeddings(query, top_k=10)
    if top_results:
        sources = "\n".join([f"Source: {src}\nContent: {txt}..." for src, txt in top_results])
        answer = ask_gemini(query, sources)
        st.markdown(f"### Answer: {answer}")
    else:
        st.write("Could not find a relevant answer.")