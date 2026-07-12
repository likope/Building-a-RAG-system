from assistant import Assistant
from judge import Judge
from eval import Eval
from embedding import Embedding

if __name__ == "__main__":  #if this file its run as main

    assistant   = Assistant()   #inizializzate classes
    judge       = Judge()
    eval        = Eval()
    embedding   = Embedding()

    history_summary     = ""    #inizializzation of variable
    answer_judge        = ""
    n_turns             = 0     #inizializzation of number of actual turns
    limit_history_turn  = 2     #inizializzation of turn limits

    while True:             #while:
        embeddings = 0
        if n_turns > limit_history_turn:    #if history is too long
            history_summary = assistant.get_history_summary()   #get a summary of the history
            n_turns = 1

        user_input = input("Insert the prompt for the LLM (or type 'exit' to quit) (or type 'reload the context'): ")
        if user_input.lower() == "exit":
            print("Program terminated!")
            break
        if user_input.lower() == "reload the context":
            vectorstore = embedding.do_embedding()
            embeddings = 1

        if embeddings == 0:
            answer_llm, current_state = assistant.Ask(user_input, history_summary, answer_judge)
            answer_judge = judge.get_evaluation(current_state)
            current_state["judge_output"] = answer_judge
            print(f"Answer from LLM:\n{answer_llm}\n")
            print(f"Answer from Judge:\n{answer_judge}\n")
            eval.evaluate_judge(current_state)
            n_turns = n_turns+1