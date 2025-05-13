from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from api.gemma3WithOllama import generate_response
from utils.text_utils import wrap_text

def setup_vectordb(file_path="contexto_RAG.csv"):
    docs = CSVLoader(file_path=file_path).load()
    split_docs = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory="./db"
    )
    return vectordb

def ayuda_mode(question, vectordb):
    retrived_docs = vectordb.similarity_search(question, k=1)
    contexto = retrived_docs[0].page_content if retrived_docs else ""
    prompt = f"""Usa el siguiente contexto para explicar al usuario de forma clara y sencilla. Si el contexto no es útil, responde de forma general:
Contexto: {contexto}

Pregunta: {question}
Respuesta:"""
    print("Ayuda gemma3:\n" + wrap_text(generate_response(prompt)))
    print(wrap_text(contexto))
