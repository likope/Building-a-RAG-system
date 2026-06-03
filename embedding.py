import client_embedding, path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


class Embedding:
    def __init__(self):
        self.embedding_model = client_embedding.embedding_model
        self.path_documents = path.path_documents
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

    def load_documents(self):
        self.documents = PyMuPDFLoader(self.path_documents / "document.pdf").load()
        return self.documents
    
    def do_embedding(self):
        self.chunks = self.splitter.split_documents(self.documents)
        self.vectorstore = FAISS.from_documents(self.chunks, self.embedding_model)