from client_assistente import params_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import client_embedding
import path


class Assistant:

    def __init__(self, path_documents = path.path_documents, embedding_model = client_embedding.embedding_model):
        self.path_documents = path_documents
        self.history = ""
        self.output = ""
        self.judge_output = ""
        self.llm = params_llm()
        self.prompt_template = """Il tuo compito è quello di rispondere nel modo piu coerente possibile all input in base alla storia della conversazione e alle indicazioni.
        input = {input}
        storia = {history}"""
        self.prompt = PromptTemplate.from_template(self.prompt_template)
    
    def get_actual_history(self, _):
        return self.history
    
    def append_history(self, user_input: str):
        self.history += f"input = {user_input}, " + f"output = {self.output}, " + f"judge_output = {self.judge_output}"
        return self.history
    
    def reset_history(self):
        self.history = ""
        return self.history

    def Ask(self, user_input: str, history_summary: str):
        print("Prompt ricevuto!\n")
        self.current_state = {
            "input": user_input,
            "history": self.history,
            "output": self.output,
            "judge_output": self.judge_output
        }
        if history_summary != "":
            self.history = history_summary
        print(f"self.history = {self.history}\n")
        print("Invokazione della catena...\n")
        self.chain = (
            RunnablePassthrough.assign(history = self.get_actual_history)
            |self.prompt
            |self.llm
            |StrOutputParser()
        )
        self.output = self.chain.invoke(self.current_state)
        self.append_history(user_input)
        self.current_state = {
            "input": user_input,
            "history": self.history,
            "output": self.output,
            "judge_output": self.judge_output
        }
        self.judge_output = self.ask_to_judge()
        self.append_history(user_input)
        return self.output, self.judge_output
    
    def get_history_summary(self):
        print("Summary della history in corso!")
        self.template_history_summary = """Il tuo compito è quello di creare un riassunto della storia di una conversazione per un altra llm.
        storia = {history}."""
        self.prompt_history_summary = PromptTemplate.from_template(self.template_history_summary)
        self.history_summary_chain = self.prompt_history_summary | self.llm | StrOutputParser()
        self.history_summary = self.history_summary_chain.invoke(self.current_state)
        print("Summary della history terminato con successo!")
        return self.history_summary