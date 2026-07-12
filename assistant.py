from client_assistant import params_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from embedding import Embedding
from operator import itemgetter

class Assistant:

    def __init__(self):
        """
        Costructor of Assistant class, which initializes the necessary attributes for the conversation with the LLM.
        """
        self.history = ""
        self.output = ""
        self.judge_output = ""
        self.llm = params_llm()
        self.prompt_template = """Your task is to respond as coherently as possible to the input, based on the history of the conversation and the context provided. Every factual statement must be accompanied by a verbatim quote from the document, enclosed in quotation marks.
        input = {input},
        history = {history},
        context = {context},"""
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        self.documents = ""
        self.embedding = Embedding()
    
    def get_actual_history(self, _):
        """
        f that returns the current conversation history.
        """
        return self.history
    
    def append_history(self, user_input: str):
        """
        f that appends the current input, output, and judge output to the conversation history.
        """
        self.history += f"input = {user_input}, " + f"output = {self.output}, " + f"judge_output = {self.judge_output}"
        return self.history
    
    def reset_history(self):
        """
        f that resets the conversation history.
        """
        self.history = ""
        return self.history
    
    def update_current_state(self, user_input: str):
        """
        f that updates the current state dictionary to be passed to the LLM.
        """

        self.current_state = {
            "input": user_input,
            "history": self.history,
            "output": self.output,
            "context": self.documents,
            "judge_output": self.judge_output
        }
    
    def get_context(self, user_input: str):
        """
        f che si occupa di restituire il contesto in base alla query dell utente.
        """
        vectorstore = self.embedding.load_vectorstore()
        docs = vectorstore.similarity_search(user_input, k=5)
        self.documents = ", ".join([doc.page_content for doc in docs])
        return self.documents
    
    def get_history_summary(self):
        """
        f that generates a summary of the conversation history to be passed to the LLM.
        """
        self.summary_llm = params_llm(temperature=0.1)
        self.template_history_summary = """Il tuo compito è quello di creare un riassunto della storia di una conversazione per un altra llm.
        history = {history}."""
        self.prompt_history_summary = PromptTemplate.from_template(self.template_history_summary)
        self.history_summary_chain = self.prompt_history_summary | self.summary_llm | StrOutputParser()
        self.history_summary = self.history_summary_chain.invoke(self.current_state)
        return self.history_summary

    def Ask(self, user_input: str, history_summary: str, answer_judge: str):
        """
        f that takes the user input, the history summary, and the judge's response as input, and returns the LLM's response and the current state.
        """
        self.judge_output = answer_judge
        self.update_current_state(user_input)

        if history_summary != "":
            self.history = history_summary

        self.chain_with_embedding = (
            RunnablePassthrough.assign(history = self.get_actual_history,
                                       context = itemgetter("input") | RunnableLambda(self.get_context))
            |self.prompt
            |self.llm
            |StrOutputParser()
        )

        self.output = self.chain_with_embedding.invoke(self.current_state)
        self.history = self.append_history(user_input)
        self.update_current_state(user_input)
        return self.output, self.current_state