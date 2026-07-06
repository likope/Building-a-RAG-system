from assistant import Assistant
from judge import Judge
from eval import Eval

if __name__ == "__main__":

    assistant  = Assistant()
    judge     = Judge()
    eval = Eval()

    history_summary = ""
    answer_judge = ""
    i = 0
    while True:

        if i > 4:
            history_summary = assistant.get_history_summary()
            i = 0

        user_input = input("Insert the prompt for the LLM (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Program terminated!")
            break

        answer_llm, current_state = assistant.Ask(user_input, history_summary, answer_judge)
        answer_judge = judge.get_evaluation(current_state)
        current_state["judge_output"] = answer_judge
        print(f"Answer from LLM:\n{answer_llm}\n")
        print(f"Answer from Judge:\n{answer_judge}\n")
        eval.evaluate_judge(current_state)
        i = i+1