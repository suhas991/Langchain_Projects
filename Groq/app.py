import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize session state for vector store
if "vector_store" not in st.session_state:
    try:
        # Initialize embeddings and document processing
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vector_store = FAISS.from_documents(
            st.session_state.final_docs,
            st.session_state.embeddings
        )
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")

# Set up the Streamlit interface
st.title("ChatGroq Demo")

# Initialize the LLM
llm = ChatGroq(
    api_key=groq_api_key,  # Updated parameter name
    model_name="llama-3.1-8b-instant",  # Updated model name
    temperature=0.7,
)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that answers questions based on the given context."),
    ("user", """
    Answer the question based only on the given context.
    Context: {context}
    Question: {input}
    """)
])

# Create the chain with output parser
document_chain = create_stuff_documents_chain(
    llm,
    prompt,
    output_parser=StrOutputParser()
)

# Create retriever and retrieval chain
if "vector_store" in st.session_state:
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Handle user input
    user_input = st.text_input("Input your prompt here")
    if user_input:
        try:
            start = time.process_time()
            response = retrieval_chain.invoke({"input": user_input})
            process_time = time.process_time() - start
            
            # Display results
            st.write(response['answer'])
            st.info(f"Response time: {process_time:.2f} seconds")
            
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"], 1):
                    st.markdown(f"**Document {i}**")
                    st.write(doc.page_content)
                    st.divider()
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
else:
    st.error("Vector store not initialized properly. Please check your configuration.")