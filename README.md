Satellite Collision Risk - LangChain RAG Assistant
This repository uses LangChain and OpenAI GPT-3.5 to predict satellite collision risks. The project involves various steps, including data generation, model training, and indexing with FAISS.


Step-by-Step Execution Order:

Run data_generator.py:
Purpose: Generates synthetic satellite collision data with added noise and labels each pair as risky or safe.

Run train_model.py:
Purpose: Trains a model based on the dataset generated in the previous step.

rag_index_builder.py:
Purpose: Generates vector embeddings for the satellite data and stores them in a FAISS index for fast retrieval.

Run rag_query.py:
Purpose: Accepts user queries, retrieves relevant data using FAISS, and passes it to GPT-3.5 for text generation.

Run rag_ui.py
Purpose: Launches the Streamlit web interface to interact with the model. Users can input queries and receive results in a web-based UI.

Project Workflow Overview:
1.data_generator.py: Generates synthetic data and saves it.

2.train_model.py: Trains the model on the generated data.

3.rag_index_builder.py: Creates vector embeddings and indexes them using FAISS.

4.rag_query.py: Handles user input, retrieves data, and generates responses using GPT-3.5.

5.rag_ui.py: Provides a web interface for user interaction.

Contributors
Nabeel Asad - Project Creator

