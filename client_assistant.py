import os
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
#from dotenv import load_dotenv
#load_dotenv()

#def params_llm(model: str = "gemini-2.5-flash", temperature: float = 0.8):
#    llm = ChatGoogleGenerativeAI(
#        model = model,
#        google_api_key = os.getenv("GOOGLE_API_KEY"),
#        temperature = temperature
#    )
#    return llm
base_url = os.getenv("OLLAMA_BASE_URL", "http://100.89.85.39:11434")

def params_llm(model: str = "deepseek-r1:8b", temperature: float = 0.8):
    llm = ChatOllama(
        model = model,
        temperature = temperature,
        base_url = base_url
    )
    return llm
