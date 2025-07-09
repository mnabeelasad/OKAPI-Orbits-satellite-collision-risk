from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import pickle

# 📥 Load saved text data (list of strings)
with open("satellite_texts.pkl", "rb") as f:
    texts = pickle.load(f)

# 🔄 Convert to LangChain Document format
documents = [Document(page_content=text) for text in texts]

# 🧠 Load SentenceTransformer embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🏗️ Create FAISS index using LangChain
vector_index = FAISS.from_documents(documents, embedding_model)

# 💾 Save LangChain-compatible FAISS index
vector_index.save_local("satellite_index")  # 👈 Creates folder with .faiss, .pkl, etc.

print("✅ LangChain-compatible FAISS index created successfully.")
