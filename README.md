# How it works:
It use ollama with deepseekR1-14B model, and bge 3 embedding model, but can change It by the file client.
The script uses the LangChain framework. First, you Need to run the main.py, next, PDF documents are loaded into a container (vector store) as context. Then, after entering a prompt, the LLM provides a response that is evaluated by the judge and a deterministic evaluation function to determine how well the LLM’s response aligns with the context and whether it has hallucinated;

# Judge Bias:
For choosing the llm, please to choose different model for assistant and judge, local LLM (Deepseek14b and Qwen7b via Ollama), explicitly trained as both a RAG assistant and a judge, systematically ignores format constraints: zero verbatim citations in 12 out of 14 cases, and no structured ratings from the judge. Prompt-based citation grounding is not a usable ground truth in this configuration.
