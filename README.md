# AI-Summer-of-Code
Three different systems that uses the Groq LLM for generating responses, streamlit for deploying the system.

For app_.py and rag_app.py also used FAISS for similarity search & Google GenAI model for embeddings. The model used for answering questions based on the context of the PDF documents is gemma2-9b-it model.

app_.py basically allows users to query pdfs loaded from a directory. 

rag_app.py allows users to query pdfs they uploaded. 

app.py and simple.py used FastAPI to interact with a Groq API using uvicorn for running and Postman to test the endpoints.
