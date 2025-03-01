import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss

# Configure Google Generative AI
genai.configure(api_key="AIzaSyCih7c0Yy5ORC766HwUeK2xuiGpgiqXt28")  
gemini = genai.GenerativeModel('gemini-1.5-flash')

# Load the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2') 

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Shehanaz_data.csv')  # Replace with your dataset file name
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("The CSV file must contain 'question' and 'answer' columns.")
            st.stop()
        # Create embeddings for all questions in the dataset
        embeddings = embedder.encode(df['question'].tolist())
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
        index.add(embeddings)
        return df, index
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

df, faiss_index = load_data()

# Streamlit UI
st.markdown('<h1>🤖 Shehanaz Shaik Clone Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<h3>Ask me anything, and I\'ll respond as Sheha!</h3>', unsafe_allow_html=True)
st.markdown("---")

# Function to find the best match using FAISS
def find_best_match(query, faiss_index, df, similarity_threshold=0.7):
    # Encode the query
    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Normalize the query embedding
    faiss.normalize_L2(query_embedding)
    
    # Search for the closest match using FAISS
    D, I = faiss_index.search(query_embedding, k=1)  # Top 1 match
    if I.size > 0:
        max_similarity = D[0][0]  # Cosine similarity score
        if max_similarity >= similarity_threshold:
            return df.iloc[I[0][0]]['answer'], max_similarity
    return None, 0

# Function to refine the answer using Gemini
def refine_answer_with_gemini(query, retrieved_answer):
    prompt = f"""You are Shehanaz, an AI Student. Respond to the following question in a friendly and conversational tone:
    Question: {query}
    Retrieved Answer: {retrieved_answer}
    - Do not add any new information.
    - Ensure the response is grammatically correct and engaging.
    """
    response = gemini.generate_content(prompt)
    return response.text

# Function to handle greetings
def handle_greeting(query):
    greetings = ["hello", "hi", "hey", "greetings", "howdy"]
    if query.lower() in greetings:
        return "Hello! How can I assist you today?"
    return None

# Chatbot logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], 
                        avatar="🙋" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        try:
            # Check if the query is a greeting
            greeting_response = handle_greeting(prompt)
            if greeting_response:
                response = f"**Sheha**:\n{greeting_response}"
            else:
                # Find the best match
                retrieved_answer, similarity_score = find_best_match(prompt, faiss_index, df, similarity_threshold=0.7)
                if retrieved_answer:
                    # Refine the answer using Gemini
                    refined_answer = refine_answer_with_gemini(prompt, retrieved_answer)
                    response = f"**Sheha**:\n{refined_answer}"
                else:
                    response = "**Sheha**:\nThis is out of context. Please ask something related to my dataset."
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
