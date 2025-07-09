import streamlit as st
import pandas as pd
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 🛰️ App Title
st.set_page_config(page_title="🛰️ Satellite Collision RAG", layout="wide")
st.title("🛰️ Satellite Collision Risk - LangChain RAG Assistant")

# 🎚️ Mode switch in sidebar
mode = st.sidebar.radio("🧠 Select Mode:", ["Simple Mode (Fast)", "RAG Mode (Detailed)"])

# 📦 Load all RAG + dataset components
@st.cache_resource
def load_rag():
    # Load FAISS vector store and satellite texts
    faiss_index = FAISS.load_local(
        "satellite_index",
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )
    with open("satellite_texts.pkl", "rb") as f:
        texts = pickle.load(f)

    df = pd.read_pickle("dataset.pkl")

    # Load OpenAI model (make sure your API key is set in env)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    # Prompt Template
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a satellite safety expert. Use the following satellite collision data to answer the user's question.

Data:
{context}

Question:
{question}

Answer in simple, clear bullet points. Do not repeat the question or explain the dataset structure.
"""
    )

    # Retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=faiss_index.as_retriever(),
        chain_type_kwargs={"prompt": custom_prompt}
    )

    return df, qa_chain

# Load RAG + data
df, qa_chain = load_rag()

# 🧠 User Input
query = st.text_input("🔍 Ask your question (e.g., 'Which satellites are risky?' or 'What is RAAN diff?')")

if query:
    query_lower = query.lower()

    if mode == "Simple Mode (Fast)":
        if any(k in query_lower for k in ['risky', 'risk', 'collision', 'danger']):
            st.subheader("🚨 Risky Satellite Pairs:")
            risky_df = df[df['risk'] == 1]
            if not risky_df.empty:
                for _, row in risky_df.sample(min(5, len(risky_df))).iterrows():
                    st.markdown(f"""
                    - 🛰️ `{row['sat1_name']}` & `{row['sat2_name']}`
                    - 📍 Distance: `{row['rel_distance']:.2f}` km  
                    - ⚡ Velocity: `{row['rel_velocity']:.2f}` km/s  
                    - 📐 Inclination Diff: `{row['inclo_diff']:.2f}` deg  
                    - 🌀 Eccentricity Diff: `{row['ecco_diff']:.6f}`  
                    - 🧭 RAAN Diff: `{row['raan_diff']:.2f}` deg  
                    """)
            else:
                st.warning("❌ No risky satellite pairs found.")

        elif "safe" in query_lower:
            st.subheader("🟢 Safe Satellite Pairs:")
            safe_df = df[df['risk'] == 0]
            if not safe_df.empty:
                for _, row in safe_df.sample(min(5, len(safe_df))).iterrows():
                    st.markdown(f"""
                    - 🛰️ `{row['sat1_name']}` & `{row['sat2_name']}`
                    - 📍 Distance: `{row['rel_distance']:.2f}` km  
                    - ⚡ Velocity: `{row['rel_velocity']:.2f}` km/s  
                    - 📐 Inclination Diff: `{row['inclo_diff']:.2f}` deg  
                    - 🌀 Eccentricity Diff: `{row['ecco_diff']:.6f}`  
                    - 🧭 RAAN Diff: `{row['raan_diff']:.2f}` deg  
                    """)
            else:
                st.warning("❌ No safe satellite pairs found.")

        else:
            st.info("ℹ️ Use keywords like 'risky' or 'safe' to filter results.")
    else:
        with st.spinner("Thinking with RAG..."):
            response = qa_chain.run(query)
        st.subheader("🧠 RAG Answer:")
        st.write(response)

# 📊 Optional: Preview some sample satellite pairs
with st.expander("📊 Show random data from satellite dataset"):
    st.dataframe(df.sample(5)[['sat1_name', 'sat2_name', 'rel_distance', 'rel_velocity', 'risk']])
