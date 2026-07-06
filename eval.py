from assistant import Assistant
import re

class Eval:
    def __init__(self):
        self.assistant = Assistant()
    
    def normalize_text(self, text: str):
        """
        f that normalizes the text by removing line breaks, hyphens, and extra spaces.
        """
        text = text.replace("-\n", "")
        text = text.replace("- ", "")
        text = text.replace("-", "")
        return " ".join(text.split())

    def evaluate(self, current_state):
        """
        f that evaluates the output of the LLM by checking if all the quoted citations in the output are present in the context. It ignores citations that contain mathematical symbols.
        """
        simboli_math = {"∫", "Γ", "√"}
        context = current_state["context"]
        output = current_state["output"]
        cit = re.findall(r'"([^"]*)"', output)
        cit_norm = [self.normalize_text(c) for c in cit]
        context_norm = self.normalize_text(context)
        for c in cit_norm:
            if any(sym in c for sym in simboli_math):
                continue
            elif c not in context_norm:
                print(f"Cit not in the context: {c}")
                return False
        print("All the cited references are present in the context.")
        return True

    def parser(self, current_state):
        """"""
        judge_output = current_state["judge_output"]
        match = re.findall(r"\$\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", judge_output)
        if match:
            accuratezza, risposta, riferimenti = match[-1]
            return accuratezza, risposta, riferimenti
        else:
            return None, None, None
        
    def evaluate_judge(self, current_state):
        """"""
        result_parser = self.parser(current_state)
        result_evaluate = self.evaluate(current_state)
        print(f"Parser result: {result_parser}, Evaluate result: {result_evaluate}")