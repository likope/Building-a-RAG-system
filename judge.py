from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from client_assistente import params_llm


class Judge:

    def __init__(self):
        """
        Costruttore statico che inizializza e crea il prompt template.
        """

        self.judge_output = ""
        self.llm = params_llm(temperature=0.1)
        self.template_judge = """Il tuo compito è quello di valutare un primo llm generatore di risposta seguendo questo template con valori assoluti tra 0 e 3: accuratezza nei confronti del contesto\n, risposta in funzione della domanda\n, capacità di fornire riferimenti in funzione del contesto.\n
        input = {input},
        llm_output = {output}
        storia = {history},
        contesto = {context}
        Devi dividere il giudizio in due parti, la prima deve essere una argomentativa, NELLA SECONDA PARTE DEVI FORNIRE UNA SOLA RIGA CHE DEVE INIZIARE OBBLIGATORIAMENTE CON QUESTO SIMBOLO: "$" DOVE CITI UN GIUDIZIO NUMERICO, SEPARATO DA UNA VIRGOLA, IN QUESTO MODO: accuratezza, risposta, riferimenti.
        """
        self.prompt_judge = PromptTemplate.from_template(self.template_judge)

    def get_evaluation(self, current_state: dict):
        """
        f che si occupa di fare valutare l attuale risposta dell llm a un llm as a judge.
        """
        
        self.judge_chain = self.prompt_judge | self.llm | StrOutputParser()
        self.judge_output = self.judge_chain.invoke(current_state)
        return self.judge_output