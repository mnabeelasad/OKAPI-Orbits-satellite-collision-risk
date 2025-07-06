import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ------------------ Load RAG Components ------------------
print("\n📥 Loading FAISS index and text data...")
index = faiss.read_index('satellite_index.faiss')
with open('satellite_texts.pkl', 'rb') as f:
    texts = pickle.load(f)

# ------------------ Load Embedding Model ------------------
print("🔍 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ Simple CLI Loop ------------------
print("\n🤖 Ask me about satellites! (type 'exit' to quit)")
while True:
    query = input("\n🧠 Your question: ")
    if query.lower() in ['exit', 'quit']: break

    # Convert question to embedding
    query_vector = model.encode([query])

    # Search in FAISS
    D, I = index.search(np.array(query_vector), k=3)  # top 3 results

    print("\n🔎 Top Relevant Results:")
    for i, idx in enumerate(I[0]):
        print(f"\n📄 Result #{i+1} (Score: {D[0][i]:.2f}):")
        print(texts[idx])

print("\n👋 Exiting RAG chat. Bye!")
