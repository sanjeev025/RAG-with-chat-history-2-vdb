from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os

DB_DIR = "faiss_index"

def create_vector_store(texts, persist=True):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([texts])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(
        chunks, 
        embeddings
    )
    
    if persist:
        vectorstore.save_local(DB_DIR)
    return vectorstore

def load_vector_store():
    if not os.path.exists(DB_DIR):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )