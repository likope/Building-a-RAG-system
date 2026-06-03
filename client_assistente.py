"""
File che contiene e gestisce il client del progetto, qui vengono gestiti i parametri e viene definita la chiamata del modello.
"""


import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

def params_llm(model: str = "gemini-2.5-flash", temperature: float = 0.8):
    llm = ChatGoogleGenerativeAI(
        model = model,
        google_api_key = os.getenv("GOOGLE_API_KEY"),
        temperature = temperature
    )
    return llm