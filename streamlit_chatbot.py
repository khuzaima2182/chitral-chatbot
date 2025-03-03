import streamlit as st
from pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import os
from langchain_pinecone import PineconeVectorStore
from google.generativeai import configure
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file

configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit UI
st.set_page_config(page_title="Chitral Travel Agent", layout="wide")
st.title("🌍 Chitral Travel Agent")
st.write("Ask me anything about Chitral! Let's plan your adventure. 🚀")

# Load API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

index_name = "travel-agent"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
if index_name not in pc.list_indexes().names():
    st.error(f"Pinecone index '{index_name}' not found. Please create it first.")
    st.stop()

# Use Langchain's built-in method to connect to an existing index
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY, index_name=index_name, embedding=embeddings)

# Set up the retriever
retriever = vectorstore.as_retriever()

# Define LangChain QA Chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "🌟 Welcome to Chitral Travel Agent! I'm here to help you explore the breathtaking beauty of Chitral. How can I assist you today?",
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get AI response
    response = qa_chain.run(user_input)

# Check if response lacks relevant information
    if any(phrase in response.lower() for phrase in [
        "not found", 
        "no information", 
        "doesn't offer any information",
        "this text doesn't mention", 
        "this text focuses", 
        "i couldn't find"
    ]):
        response = (
            "Hmm, I couldn't find any details on that right now. 🤔 "
            "But don't worry, I'm always learning and updating! "
            "Feel free to ask about something else, or check back later for more info. 😊"
        )

    # Enhance response with storytelling
    response = f" {response}"
    
    # Append AI response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display updated chat
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response)
