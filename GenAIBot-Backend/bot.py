from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize the chat model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768"
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Vector store and documents paths
VECTOR_STORE_PATH = "vector_store"
DOCUMENTS_PATH = "documents"

# Initialize vector store
def initialize_vector_store():
    # Create documents directory if it doesn't exist
    os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
    else:
        # Load any existing documents from the documents directory
        if os.path.exists(DOCUMENTS_PATH) and os.listdir(DOCUMENTS_PATH):
            loader = DirectoryLoader(
                DOCUMENTS_PATH,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            texts = text_splitter.split_documents(documents)
        else:
            texts = []

        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
        if texts:
            vector_store.add_documents(texts)
        vector_store.persist()
    return vector_store

# Initialize the conversation chain
def initialize_chain():
    vector_store = initialize_vector_store()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 3}
        ),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    return chain

# Initialize the chain
qa_chain = initialize_chain()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('userQuery', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Get response from the chain
        response = qa_chain({"question": user_message})
        
        # Format the response
        answer = response['answer']
        sources = []
        if 'source_documents' in response:
            sources = [doc.metadata.get('source', 'Unknown') for doc in response['source_documents']]
            sources = list(set(sources))  # Remove duplicates

        return jsonify({
            "choices": [{
                "message": {
                    "content": answer,
                    "sources": sources
                }
            }]
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_documents():
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist('files')
        
        # Create documents directory if it doesn't exist
        os.makedirs(DOCUMENTS_PATH, exist_ok=True)

        # Save uploaded files to documents directory
        saved_files = []
        for file in files:
            if file.filename:
                filepath = os.path.join(DOCUMENTS_PATH, file.filename)
                file.save(filepath)
                saved_files.append(filepath)

        # Process documents
        documents = []
        for file_path in saved_files:
            loader = TextLoader(file_path)
            documents.extend(loader.load())

        # Split documents
        texts = text_splitter.split_documents(documents)

        # Update vector store
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
        vector_store.add_documents(texts)
        vector_store.persist()

        # Reinitialize the chain with updated vector store
        global qa_chain
        qa_chain = initialize_chain()

        return jsonify({
            "message": f"Successfully processed {len(files)} files. Documents are stored in the 'documents' directory."
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear-chat', methods=['DELETE'])
def clear_chat():
    try:
        # Clear conversation memory
        memory.clear()
        
        return jsonify({"message": "Chat history cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=4444, debug=True)