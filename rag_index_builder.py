import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# STEP 1: Load your dataset
with open('dataset.pkl', 'rb') as f:
    df = pickle.load(f)

# STEP 2: Convert rows to plain text
texts = []
for index, row in df.iterrows():
    text = (
        f"Satellite Pair: {row['sat1_name']} & {row['sat2_name']}. "
        f"Relative distance is {row['rel_distance']:.2f} km, "
        f"Relative velocity is {row['rel_velocity']:.2f} km/s. "
        f"Inclination diff: {row['inclo_diff']:.2f} deg, "
        f"Eccentricity diff: {row['ecco_diff']:.6f}, "
        f"RAAN diff: {row['raan_diff']:.2f} deg. "
        f"Collision Risk: {'Yes' if row['risk'] == 1 else 'No'}."
    )
    texts.append(text)

# STEP 3: Load SentenceTransformer model
print("\n🔍 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# STEP 4: Generate embeddings
print("🔁 Creating vector embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# STEP 5: Store in FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# STEP 6: Save FAISS & metadata
faiss.write_index(index, 'satellite_index.faiss')
with open('satellite_texts.pkl', 'wb') as f:
    pickle.dump(texts, f)

print("\n✅ FAISS index and text data saved.")
