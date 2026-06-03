"""
File che contiene i parametri del modello di embedding.
"""


from langchain_ollama import OllamaEmbeddings

embedding_model = OllamaEmbeddings(model="bge-m3", truncate="END")