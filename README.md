# How it works:
It uses ollama with deepseekR1-8B model, and bge 3 embedding model, but can be changed in the client files.
**To use the script:**
- please install Ollama with the desired models and start the Ollama server;
- next clone the repo and run on terminal "pip install -r requirements.txt" to install all the dependencies,
- now you can run the script on terminal with "python main_but_gradio.py".

# Features:
- LLM Assistant;
- LLM as Judge;
- Vectorstore;
- Offline, the script runs fully offline on the host machine, no api costs and no data leaving the environment;

# Research:
I'm working on research with the self-preference bias when the assistant and judge have the same model, for this I've built a ground-truth function that verifies the verbatim citations of assistant answer and context and the judge exhibited bias in 12 of 14 runs, please choose different model for assistant and judge.
