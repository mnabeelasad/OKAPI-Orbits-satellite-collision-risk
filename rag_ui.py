import streamlit as st
import pickle
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model and index
st.title("🛰️ Satellite Collision Risk - RAG Assistant")

@st.cache_resource
def load_rag_components():
    df = pd.read_pickle("dataset.pkl")
    index = faiss.read_index("satellite_index.faiss")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return df, index, model

df, index, model = load_rag_components()

# Input from user
query = st.text_input("🔎 Ask a question (e.g., 'Which satellites are risky?')")

if query:
    user_query = query.strip().lower()
    
    # Generate embedding
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding).astype('float32'), k=10)  # Get more in case we filter

    results = [df.iloc[row_idx] for row_idx in I[0]]
    
    if user_query == "risky":
        filtered = [row for row in results if row['risk'] == 1]
        st.subheader("🔴 Risky Satellite Pairs")
    elif user_query == "safe":
        filtered = [row for row in results if row['risk'] == 0]
        st.subheader("🟢 Safe Satellite Pairs")
    else:
        filtered = results
        st.subheader("📊 Top Relevant Results")

    if filtered:
        for i, row in enumerate(filtered, 1):
            st.markdown(f"""
            **📄 Result #{i}:**
            - 🛰️ Satellite Pair: `{row['sat1_name']}` & `{row['sat2_name']}`
            - 📍 Relative Distance: `{row['rel_distance']:.2f}` km
            - 📍 Relative Velocity: `{row['rel_velocity']:.2f}` km/s
            - 📍 Inclination Diff: `{row['inclo_diff']:.2f}` deg
            - 📍 Eccentricity Diff: `{row['ecco_diff']:.6f}`
            - 📍 RAAN Diff: `{row['raan_diff']:.2f}` deg
            - 🚨 **Collision Risk**: `{"Yes" if row['risk'] == 1 else "No"}`
            """)
    else:
        st.warning("❌ Koi matching result nahi mila.")
