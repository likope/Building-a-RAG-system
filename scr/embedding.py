from client.client_embedding import embedding_model
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from path import path_base, path_documents, path_for_vs


class Embedding:
    def __init__(self):
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.path_documents = path_documents
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=250)

    def load_documents(self):
        documents = []
        for pdf_path in sorted(self.path_documents.glob("*.pdf")):
            print(f"Upload file: {pdf_path}")
            documents.extend(PyMuPDFLoader(pdf_path).load())
        if documents is None or len(documents) == 0:
            print("No documents found in the 'documents' directory. Please add PDF files to the 'documents' directory.")
            return None
        return documents

    def save_vectorstore(self, vectorstore):
        vectorstore.save_local(path_base)
        print(f"Vectorstore saved in: {path_base}/vectorstore")

    def load_vectorstore(self):
        if self.vectorstore is not None:
            print("Vectorstore already loaded.")
            return self.vectorstore
        try:
            vectorstore = FAISS.load_local(path_for_vs, self.embedding_model, allow_dangerous_deserialization=True)
        except (RuntimeError, ValueError, OSError) as e:
            print(f"Vectorstore non caricato da {path_for_vs}: {e}")
            path_for_vs.mkdir(parents=True, exist_ok=True)
            print(f"Vectorstore directory created: {path_for_vs}")
            return 2
        self.vectorstore = vectorstore
        return vectorstore

    def do_embedding(self):
        docs = self.load_documents()
        if docs is None:
            print("No documents found in the 'documents' directory. Please add PDF files to the 'documents' directory.")
            return None  # noqa: RET501
        print("start embedding")
        chunks = self.splitter.split_documents(docs)
        chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
        print(f"[ingest] valid chunck: {len(chunks)}")
        vectorstore = None
        for i in range(0, len(chunks), 16):
            batch = chunks[i : i + 16]
            print(f"[ingest] batch {i}-{i + len(batch)}")
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, self.embedding_model)
            else:
                vectorstore.add_documents(batch)
        print("Done!")
        self.save_vectorstore(vectorstore, path_for_vs)
        self.vectorstore = vectorstore
