#definisco una f che prende l input e output e costruisce la history relativa nel state
def update_history(state: dict) -> str:
    return state["history"] + f"User input: {state['input']}, Assistant output: {state['output']}"