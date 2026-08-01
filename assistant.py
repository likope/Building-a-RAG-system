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
        self.llm = params_llm()
        self.prompt_template = """You are a LLM, you have to answer properly based on the user input, you can look history of your conversation and I'll give you the context that the user want your answer are based.\n
        If context = 'NO DOCS RELEVANT IN CONTEXT' is because there aren't properly context on this user input, say at the user that there isn't properly context.\n
        input = {input},
        history = {history},
        context = {context},"""
        self.prompt = PromptTemplate.from_template(self.prompt_template)
        self.embedding = Embedding()
    
    def get_actual_history(self, _):
        """
        f that returns the current conversation history.
        """
        return self.history
    
    def append_history(self, user_input: str, output: str = "", judge_output: str = ""):
        """
        f that appends the current input, output, and judge output to the conversation history.
        """
        self.history += f"input = {user_input}, " + f"output = {output}, " + f"judge_output = {judge_output}"
        return self.history
    
    def reset_history(self):
        """
        f that resets the conversation history.
        """
        self.history = ""
        return self.history
    
    def update_current_state(self, user_input: str, output: str = "", judge_output: str = "", documents: str = ""):
        """
        f that updates the current state dictionary to be passed to the LLM.
        """

        current_state = {
            "input": user_input,
            "history": self.history,
            "output": output,
            "context": documents,
            "judge_output": judge_output
        }
        return current_state
    
    def get_context(self, user_input: str):
        """
        f che si occupa di restituire il contesto in base alla query dell utente.
        """
        vectorstore = self.embedding.load_vectorstore()
        docs = vectorstore.similarity_search_with_score(user_input, k=5)
        for doc, score in docs:
            print(f"score={score:.3f}  |  {doc.page_content[:80]}")
        min_score = 1.15
        docs_relevant = [doc for doc, score in docs if score < min_score]
        if not docs_relevant:
            documents = "NO DOCS RELEVANT IN CONTEXT"
            return documents
        documents = ", ".join([doc.page_content for doc in docs_relevant])
        return documents
    
    def get_history_summary(self):
        """
        f that generates a summary of the conversation history to be passed to the LLM.
        """
        self.summary_llm = params_llm(temperature=0.1)
        self.template_history_summary = self.template_history_summary = """You are summarizing a conversation history for another LLM.
        Write a brief summary that captures what the user asked and what was answered, IN YOUR OWN WORDS.
        Do NOT include any direct quotes, quoted text, or page references from the documents.
        Report only the meaning of the exchange, never the exact wording of any cited passage.
        history = {history}"""
        self.prompt_history_summary = PromptTemplate.from_template(self.template_history_summary)
        history_summary_chain = self.prompt_history_summary | self.summary_llm | StrOutputParser()
        history_summary = history_summary_chain.invoke({"history": self.history})
        return history_summary

    def Ask(self, user_input: str, history_summary: str, answer_judge: str):
        """
        f that takes the user input, the history summary, and the judge's response as input, and returns the LLM's response and the current state.
        """
        output = ""
        documents = self.get_context(user_input)
        judge_output = answer_judge
        current_state = self.update_current_state(user_input, output, judge_output, documents)

        if history_summary != "":
            self.history = history_summary
            
        chain_with_embedding = (
            RunnablePassthrough.assign(history = self.get_actual_history)
            |self.prompt
            |self.llm
            |StrOutputParser()
        )

        output = chain_with_embedding.invoke(current_state)
        self.history = self.append_history(user_input, output, judge_output)
        current_state = self.update_current_state(user_input, output, judge_output, documents)
        return output, current_state, documents