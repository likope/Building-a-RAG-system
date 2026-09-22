import os

from langchain_ollama import ChatOllama

base_url = os.getenv("OLLAMA_BASE_URL", "http://100.87.101.61:11434")


def params_llm(model: str = "deepseek-r1:8b", temperature: float = 0.8):
    llm = ChatOllama(model=model, temperature=temperature, base_url=base_url)
    return llm
