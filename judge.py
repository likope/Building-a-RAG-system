from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from client_assistant import params_llm


class Judge:

    def __init__(self):
        """
        constructor of Judge class, which initializes the necessary attributes for the evaluation of the LLM's response.
        """
        self.llm = params_llm(temperature=0.1)
        self.template_judge = """Your task is to evaluate the response of the LLM based on the input, output, history of the conversation, and context provided. You must provide a detailed argumentation for your evaluation. The evaluation should be divided into two parts: the first part should be argumentative, and the second part should consist of a single line that starts with the symbol "$" and contains a numerical judgment in scale 0-3 (only int) in this EXACT format: '$x,y,z' where: x=judge for accuracy,y=judge for faithfulness,z=judge for completeness.\n
        Here an example of judge: '$2,1,3' where 2 for accuracy, 1 for faithfulness and 3 for completeness.
        Do not treat the history of conversation like the context, but like the past conversation by user and llm.\n
        input = {input},\n
        llm_output = {output},\n
        history of the conversation= {history},\n
        context = {context},\n
        """
        self.prompt_judge = PromptTemplate.from_template(self.template_judge)

    def get_evaluation(self, current_state: dict):
        """
        f that is responsible for evaluating the current response of the LLM by a judge LLM.
        """
        
        judge_chain = self.prompt_judge | self.llm | StrOutputParser()
        judge_output = judge_chain.invoke(current_state)
        return judge_output