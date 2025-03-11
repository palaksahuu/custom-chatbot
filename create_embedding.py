from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load scraped data
with open("data.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

# Use Hugging Face embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_texts([text_data], embedding_model)

# Save the vector database
vector_db.save_local("faiss_index")
print(" Embeddings Stored Successfully!")
