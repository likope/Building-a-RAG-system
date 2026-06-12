from client_embedding import embedding_model
from path import path_documents
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


class Embedding:
    def __init__(self):
        self.embedding_model = embedding_model
        self.path_documents = path_documents
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    def load_documents(self):
        documents = []
        for pdf_path in sorted(self.path_documents.glob("*.pdf")):
            print(f"Carico il file: {pdf_path}")
            documents.extend(PyMuPDFLoader(pdf_path).load())
        return documents
    
    def do_embedding(self):
        docs = self.load_documents()
        print("inizio embedding")
        chunks = self.splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        print("Fatto Embedding")
        return vectorstore