from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from client_assistente import params_llm


class Judge:

    def __init__(self):
        """
        constructor of Judge class, which initializes the necessary attributes for the evaluation of the LLM's response.
        """

        self.judge_output = ""
        self.llm = params_llm(temperature=0.1)
        self.template_judge = """Your task is to evaluate the response of the LLM based on the input, output, history, and context provided. You must provide a detailed argumentation for your evaluation. The evaluation should be divided into two parts: the first part should be argumentative, and the second part should consist of a single line that starts with the symbol "$" and contains a numerical judgment in the format: accuracy, response, references.\n
        input = {input},
        llm_output = {output},
        history = {history},
        context = {context},
        """
        self.prompt_judge = PromptTemplate.from_template(self.template_judge)

    def get_evaluation(self, current_state: dict):
        """
        f that is responsible for evaluating the current response of the LLM by a judge LLM.
        """
        
        self.judge_chain = self.prompt_judge | self.llm | StrOutputParser()
        self.judge_output = self.judge_chain.invoke(current_state)
        return self.judge_output