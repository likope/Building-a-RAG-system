from langchain_ollama import OllamaEmbeddings
import os


base_url = os.getenv("OLLAMA_BASE_URL", "http://100.89.85.39:11434")
embedding_model = OllamaEmbeddings(model="bge-m3",
                                   base_url = base_url)