# How it works:
It use ollama with deepseekR1-14B model, and bge 3 embedding model, but can change It by the file client.
The script uses the LangChain framework. First, you Need to run the main.py, next, PDF documents are loaded into a container (vector store) as context. Then, after entering a prompt, the LLM provides a response that is evaluated by the judge and a deterministic evaluation function to determine how well the LLM’s response aligns with the context and whether it has hallucinated;

# Coming soon, 
I plan to measure how often the judge gives themselves too high a score when the LLM and the judge are the same; this can be done by comparing the judge’s scores with those of the deterministic evaluation.

- Judge bias: In my runs, I’ve found that when the judge uses the same model as the LLM, they increase the reliability scores even when the deterministic evaluation flags the response as unreliable.
- Deterministic evaluation: This compares the citations in the LLM’s response to the context’s content. It is an objective metric, but I encountered issues normalizing the text from PDF to a Python string, which could lead to citation errors. For now, I have chosen to *exclude mathematical formulas* from the citations due to the complexity of detecting them.
