# src/rag_utils.py
# Reusable utilities for RAG pipelines across projects
# Author: Your Name | Date: January 2026

import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_churn_data(csv_path: str) -> pd.DataFrame:
    """
    Load and do minimal cleaning on Telco churn dataset.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    df = df.dropna(subset=['totalcharges'])
    return df


def create_customer_profiles(df: pd.DataFrame) -> list[Document]:
    """
    Convert DataFrame rows into natural language customer profiles
    suitable for RAG context.
    """
    docs = []
    for _, row in df.iterrows():
        profile_text = (
            f"Customer Profile: "
            f"Senior Citizen: {'Yes' if row['seniorcitizen'] else 'No'}, "
            f"Partner: {row['partner']}, Dependents: {row['dependents']}, "
            f"Tenure: {row['tenure']} months, Contract: {row['contract']}, "
            f"Internet Service: {row['internetservice']}, "
            f"Monthly Charges: ${row['monthlycharges']:.2f}, "
            f"Total Charges: ${row['totalcharges']:.2f}, "
            f"Payment Method: {row['paymentmethod']}, "
            f"Churn: {row['churn']}"
        )
        doc = Document(page_content=profile_text, metadata=row.to_dict())
        docs.append(doc)
    return docs


def build_vector_store(documents: list[Document], embedding_model: str = "BAAI/bge-base-en-v1.5"):
    """
    Chunk documents, embed, and build FAISS vector store.
    """
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # Vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def query_rag(vector_store, query: str, k: int = 5):
    """
    Simple retrieval + return relevant docs (to be used with LLM chain later).
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.invoke(query)
    return relevant_docs