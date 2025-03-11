from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)

# Load stored embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get("query")
    
    # Perform similarity search in the vector database
    results = vector_db.similarity_search(user_query, k=3)
    
    response_text = "\n".join([res.page_content for res in results])
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)
