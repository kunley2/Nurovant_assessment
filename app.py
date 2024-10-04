from flask import Flask, request, jsonify
import os
import uuid
import sqlite3
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
import faiss
from src.helper import answer_query, evaluate_answer

app = Flask(__name__)

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

    # file.save(file.filename)
    data = file.read().decode('utf-8', errors='ignore')
    # loader = PyPDFLoader(file.filename)
    # data = loader.load()
    # print('content',data)

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(data)

    # Create embeddings and build vectorstore
    embeddings = HuggingFaceEmbeddings()
    # index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    global vectorstore
    vectorstore = FAISS.from_documents(
        documents=texts,
        embedding=embeddings
    )

    return jsonify({'message': 'Document uploaded and processed successfully.'}), 200


@app.route('/query/', methods=['POST'])
def query():
    if vectorstore is None:
        return jsonify({'error': 'No file provided.'}), 400
    question = request.json['question']
    answer, test_question, bullet_points = answer_query(input=question,vector_store=vectorstore)
    test_question_id = str(uuid.uuid4())

    # Store test question and answer in the database
    test_answer = answer  # For simplicity, you can generate a different test answer
    cursor.execute(
        '''
        INSERT INTO test_answers (id, test_question, test_answer) VALUES (?, ?, ?)
        ''', (test_question_id, test_question, test_answer)
    )
    conn.commit()

    response = {
        "answer": answer,
        "bullet_points": bullet_points,
        "test_question": test_question,
        "test_question_id": test_question_id
    }
    return jsonify(response)

@app.route('/evaluate/', methods=['POST'])
def evaluate():
    user_answer = request.json['user_answer']
    test_question_id = request.json['test_question_id']

    cursor.execute('SELECT test_answer FROM test_question WHERE id=?', (test_question_id,))
    row = cursor.fetchone()

    if row is None:
        return jsonify({'error': 'Test question not found.'}), 404

    question = row[0]

    knowledge, confidence = evaluate_answer(question, user_answer, vectorstore)
    response = {
        "knowledge_understood": knowledge,
        "knowledge_confidence": confidence
    }
    return jsonify(response)






if __name__ == "__main__":
    app.run(debug=True)