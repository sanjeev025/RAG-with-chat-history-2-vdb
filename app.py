import streamlit as st
from utils import extract_text_from_pdf
from rag_chain import get_rag_response
from chroma_vdb import create_vector_store
# from faiss_vdb import create_vector_store
import json
from datetime import datetime

st.set_page_config(page_title="RAG with Chat History", layout="wide")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

st.title("🔍 RAG with Chat History")

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        create_vector_store(text)
        st.session_state.document_uploaded = True
        st.success("✅ Document embedded!")
    
    # Chat statistics and export
    st.header("📊 Chat Statistics")
    if st.session_state.chat_history:
        st.metric("Total Questions", len(st.session_state.chat_history))
        if st.button("💾 Save Chat History"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.chat_history, f, indent=2, ensure_ascii=False)
            st.success(f"💾 Saved as {filename}")
    else:
        st.info("No conversations yet. Upload a document and start asking questions!")

# Main chat interface
st.header("💬 Chat with Your Document")

# Display chat history
if st.session_state.chat_history:
    st.subheader("📜 Previous Conversations")
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.info(f"**Q{i+1}:** {chat['question']}")
            with col2:
                st.success(f"**A{i+1}:** {chat['answer']}")
        st.divider()

# Question input with form to prevent auto-rerun
with st.form("question_form", clear_on_submit=True):
    question = st.text_input("Enter your question:", key="question_input")
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.form_submit_button("🚀 Ask", use_container_width=True)
    with col2:
        clear_button = st.form_submit_button("🗑️ Clear Chat History", use_container_width=True)

# Handle clear button
if clear_button:
    st.session_state.chat_history = []
    st.success("🗑️ Chat history cleared!")
    st.rerun()

# Handle question submission
if ask_button and question:
    if st.session_state.document_uploaded:
        # Format previous chat history for context
        chat_context = ""
        for chat in st.session_state.chat_history[-3:]:  # Use last 3 conversations for context
            chat_context += f"Q: {chat['question']}\nA: {chat['answer']}\n\n"
        
        # Get response
        with st.spinner("🤔 Thinking..."):
            response = get_rag_response(question, chat_context)
        
        # Display current answer
        st.write("### 📌 Current Answer:")
        st.success(response)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Rerun to refresh the display
        st.rerun()
    else:
        st.error("📄 Please upload a document first!")

# Show message if user clicks Ask without entering a question
elif ask_button and not question:
    st.warning("⚠️ Please enter a question first!")
