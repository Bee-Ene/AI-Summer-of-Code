import os
import time
import io
import PyPDF2
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  # vector store
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # vector embedding

from dotenv import load_dotenv

load_dotenv()

# Loading groq and Google API key
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Setting title
st.title('A SIMPLE MULTIPLE PDFS UPLOAD RAG USING GEMMA MODEL')

# Calling LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name='gemma2-9b-it')

# Setting prompt template
prompt = ChatPromptTemplate.from_template(
    '''
    Answer the following questions accurately based on the context.
    <context>
    {context}
    <context>
    Questions: {input}
    '''
)

# Custom Document class for handling raw text
class RawTextDocument:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata 
        
#PDF uploads
def pdf_upload(uploaded_files):
    pdf_texts = []
    for uploaded_file in uploaded_files:
        with io.BytesIO(uploaded_file.read()) as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            text = ''
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
            pdf_texts.append(text)
    return pdf_texts

# Vector embeddings
def vector_embeddings(docs):
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.docs_final = st.session_state.text_splitter.split_documents(docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.docs_final, st.session_state.embeddings)

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.write("Files uploaded successfully.")
    # Process uploaded PDFs
    pdf_texts = pdf_upload(uploaded_files)
    # Combine text from all PDFs into one list of documents
    documents = [RawTextDocument(text, metadata={'source': f'PDF {i+1}'}) for i, text in enumerate (pdf_texts)]
    
    # Initialize vector embeddings with uploaded documents
    vector_embeddings(documents)
    
    st.write("Vector embeddings are ready. You can now enter your query.")

    # Prompt
    prompt_ = st.text_input('Initialize your query first by clicking Create Query.')

    if st.button('Create query'):
        doc_chain = create_stuff_documents_chain(llm, prompt)
        retrieval = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retrieval, doc_chain)
        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': prompt_})
        st.write(response['answer'])
        # Streamlit expander to find relevant chunks
        with st.expander('Document Similarity Search'):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------')
else:
    st.write("Please upload PDF files to get started.")