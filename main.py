from llm import GeminiLlm

gemini = GeminiLlm()
response = gemini.invoke("oggi che giorno è?")
print(response)