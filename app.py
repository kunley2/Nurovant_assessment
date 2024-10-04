from flask import Flask, request, jsonify
import os
import uuid
import sqlite3
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
import faiss
from src.helper import answer_query

app = Flask(__name__)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = 'your-openai-api-key'

# Initialize global variables
vectorstore = None
chat_history = []

# Initialize SQLite database
conn = sqlite3.connect('test_answers.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS test_answers (
        id TEXT PRIMARY KEY,
        test_question TEXT,
        test_answer TEXT
    )
''')
conn.commit()


@app.route('/upload/', methods=['POST'])
def upload_document():
    file = request.files.get('file')

    if not file:
        return jsonify({'error': 'No file provided.'}), 400

    content = file.read().decode('utf-8')

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(content)

    # Create embeddings and build vectorstore
    embeddings = HuggingFaceEmbeddings()
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    global vectorstore
    vectorstore = FAISS.from_documents(
        documents=texts,
        embedding=embeddings
    )

    return jsonify({'message': 'Document uploaded and processed successfully.'}), 200
