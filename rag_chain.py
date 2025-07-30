from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from dotenv import load_dotenv
import google.generativeai as genai
import os
from typing import Any, List, Optional
from chroma_vdb import load_vector_store
# from faiss_vdb import load_vector_store

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

class GeminiLLM(LLM):
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.0
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text

def build_rag_chain():
    vectorstore = load_vector_store()
    if vectorstore is None:
        return None
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    llm = GeminiLLM(model_name="gemini-1.5-flash", temperature=0)
    
    # Define system prompt template with chat history
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    You can also reference the previous conversation history to provide better context-aware answers.

    Previous conversation history:
    {chat_history}
    
    Current context from documents: {context}

    Current question: {question}
    
    Answer: """
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question", "chat_history"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

def get_rag_response(question, chat_history=""):
    """Get response from RAG chain with chat history context"""
    rag_chain = build_rag_chain()
    if not rag_chain:
        return "Please upload a document first."
    
    # Format chat history for the prompt
    formatted_history = chat_history if chat_history else "No previous conversation."
    
    # Get relevant documents
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create the full prompt
    prompt = f"""Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    You can also reference the previous conversation history to provide better context-aware answers.

    Previous conversation history:
    {formatted_history}
    
    Current context from documents: {context}

    Current question: {question}
    
    Answer: """
    
    # Get response from LLM
    llm = GeminiLLM(model_name="gemini-1.5-flash", temperature=0)
    response = llm._call(prompt)
    
    return response
