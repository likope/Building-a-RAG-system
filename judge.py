import main
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class Judge:
    def __init__(self):
        self.judge_output = ""

    def ask_to_judge(self):
        print("Il judge sta valutando...\n")
        self.template_judge = """Il tuo compito è quello di valutare un primo llm generatore di risposta, devi essere OGGETTIVO e SEVERO, per poi dare un valore assoluto a diversi elementi della risposta da 1 a 5; spiegane anche il motivo.
        input = {input},
        llm_output = {output}
        storia = {history},"""
        self.prompt_judge = PromptTemplate.from_template(self.template_judge)
        self.judge_chain = self.prompt_judge | self.llm | StrOutputParser()
        self.judge_output = self.judge_chain.invoke(self.current_state)
        print("Valutazione terminata con successo!")
        return self.judge_output