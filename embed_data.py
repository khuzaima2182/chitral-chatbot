import os
import firebase_admin
from firebase_admin import credentials, firestore
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorstore
from langchain.schema import Document

# Set API keys
os.environ["PINECONE_API_KEY"] = "pcsk_JWfDU_QZB4ypC4J6SxQzpjE7B1X3nMftTd33roK1XsH3BPtPhpvT9qjuek7whACLh4CtR"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDcfQFFM_PEZxJTxH5KmoZANch0qMvZ2VE"

EMBEDDING_DIMENSION = 768  

# Load API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FIREBASE_CRED_PATH = "chitral-travel-agent-firebase-adminsdk-fbsvc-609c3382da.json"

# Validate API keys
if not PINECONE_API_KEY:
    raise ValueError("❌ Pinecone API Key is missing! Set it as an environment variable.")
if not GOOGLE_API_KEY:
    raise ValueError("❌ Google API Key is missing! Set it as an environment variable.")

# Initialize Firebase
cred = credentials.Certificate(FIREBASE_CRED_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define index name
index_name = "travel-agent"

# Check if index exists and has the correct dimension
existing_indexes = pc.list_indexes()

if index_name in existing_indexes.names():
    index_info = pc.describe_index(index_name)
    if index_info.dimension != EMBEDDING_DIMENSION:
        print(f"⚠️ Index '{index_name}' exists but has incorrect dimension {index_info.dimension}. Deleting...")
        pc.delete_index(index_name)  # ✅ Delete old index
        print(f"🗑️ Deleted index '{index_name}'.")

# Now create a new index with the correct dimension
if index_name not in pc.list_indexes().names():
    print(f"ℹ️ Creating index '{index_name}' with dimension {EMBEDDING_DIMENSION}...")
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,  # ✅ Correct dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Created index '{index_name}' with dimension {EMBEDDING_DIMENSION}.")

# Connect to the index
index = pc.Index(index_name)

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Define categories from Firestore
categories = ["places", "hotels", "activities", "restaurants", "transportation", "festivals", "itineraries"]

docs = []
for category in categories:
    entries = db.collection(category).stream()
    
    for entry in entries:
        data = entry.to_dict()
        doc = Document(
            page_content=data.get("description", "No description available"),  
            metadata={"name": data.get("name", "Unknown"), "category": category}
        )
        docs.append(doc)

if docs:
    # Store embeddings in Pinecone
    vectorstore = PineconeVectorstore.from_documents(docs, embeddings, index_name=index_name)
    print("✅ Embeddings successfully stored in Pinecone!")
else:
    print("⚠️ No documents found in Firebase. Skipping embedding.")
