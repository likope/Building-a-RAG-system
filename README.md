# How it works:
It uses ollama with deepseekR1-8B model, and bge 3 embedding model, but can be changed in the client files.
**To use the script:**
- please install Ollama;
- for install the LLM model: "ollama pull <llm_name>" -> the llm can be search via the official site of ollama;
- for this script is required an LLM and a embedded model, the process for install the embedder is the same as for llm;
- next clone the repo and run on terminal "pip install -r requirements.txt" to install all the dependencies;
- now you can run the script on terminal with "python main_with_gradio.py".
- **This repo is under constant update**

# Features:
- LLM Assistant;
- LLM as Judge;
- Vectorstore;
- Contanaizer;
- FastAPI;
- Now this repo can be hosted on server and work with curl command;

# Research:
I'm working on research with the self-preference bias when the assistant and judge have the same model, for this I've built a ground-truth function that verifies the verbatim citations of assistant answer and context and the judge exhibited bias in 12 of 14 runs, please choose different model for assistant and judge.

# Update:
- The model of the various LLM can be select via request POST with the variable "model";
