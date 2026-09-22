import os

from langchain_ollama import OllamaEmbeddings

base_url = os.getenv("OLLAMA_BASE_URL", "http://100.87.101.61:11434")
embedding_model = OllamaEmbeddings(model="bge-m3", base_url=base_url)
