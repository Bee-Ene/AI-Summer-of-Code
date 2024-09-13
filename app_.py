import os 
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS #vector store
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #vector embedding 

from dotenv import load_dotenv

load_dotenv()

#loading groq and google API key 
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

#setting title
st.title('A SIMPLE RAG USING GEMMA MODEL')

#calling llm
llm = ChatGroq(groq_api_key = groq_api_key, model_name = 'gemma2-9b-it')

#setting prompt template
prompt = ChatPromptTemplate.from_template(
    '''
    Answer the following questions accurately based on the context.
    <context>
    {context}
    <context>
    Questions: {input}
    
    '''
)

#vector embeddings
def vector_embeddings():
    if 'vectors' not in st.session_state:
        #Using Google GenAI embeddings
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
        #taking the data
        st.session_state.loader = PyPDFDirectoryLoader('./pdfs')
        #loading docs
        st.session_state.docs = st.session_state.loader.load()
        #splitting the data
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.docs_final = st.session_state.text_splitter.split_documents(st.session_state.docs)
        #vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.docs_final, st.session_state.embeddings)
        

#prompt
prompt_ = st.text_input('Initialize your query first by clicking Create Query.')

if st.button('Create query'):
    vector_embeddings()
    st.write('Query is ready. \n You can now enter what you want to inquire about from the documents.')
    

if prompt_:
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retrieval, doc_chain)
    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': prompt_})
    st.write(response['answer'])
    #streamlit expander to find relevant chunks
    with st.expander('Document Similarity Search'):
        for i, doc, in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------')