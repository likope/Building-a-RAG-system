from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from client_assistente import params_llm


class Judge:

    def __init__(self):
        """
        Costruttore statico che inizializza e crea il prompt template.
        """

        self.judge_output = ""
        self.llm = params_llm()
        self.template_judge = """Il tuo compito è quello di valutare un primo llm generatore di risposta, devi essere OGGETTIVO e SEVERO, per poi dare un valore assoluto a diversi elementi della risposta da 1 a 5; spiegane anche il motivo.
        input = {input},
        llm_output = {output}
        storia = {history},"""
        self.prompt_judge = PromptTemplate.from_template(self.template_judge)

    def get_evaluation(self, current_state: dict):
        """
        f che si occupa di fare valutare l attuale risposta dell llm a un llm as a judge.
        """
        
        self.judge_chain = self.prompt_judge | self.llm | StrOutputParser()
        self.judge_output = self.judge_chain.invoke(current_state)
        return self.judge_output