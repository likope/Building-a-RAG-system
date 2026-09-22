from assistant import Assistant
from embedding import Embedding
from eval import Eval
from judge import Judge


class Main:
    def __init__(self):
        self.limit_history_turn = 0
        self.n_turns = 0
        self.history_summary = ""
        self.judge_answer = ""
        self.assistant = Assistant()
        self.judge = Judge()
        self.eval = Eval()
        self.embedding = Embedding()

    def reload_context(self):
        self.embedding.do_embedding()
        self.assistant.embedding.vectorstore = None
        return "Context reloaded."

    def run_turn(self, user_input):
        # if number of turns exceeds the limit, do a summary of the history
        if self.n_turns > self.limit_history_turn:
            self.history_summary = self.assistant.get_history_summary()
            self.n_turns = 1
        answer_llm, current_state, documents = self.assistant.Ask(user_input, self.history_summary, self.judge_answer)
        self.history_summary = ""
        answer_judge = self.judge.get_evaluation(current_state)
        current_state["judge_output"] = answer_judge
        self.eval.evaluate_judge(current_state)
        self.judge_answer = answer_judge
        self.n_turns += 1
        return answer_llm, answer_judge, documents
