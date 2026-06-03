"""
File main dove viene eseguito il programma, qui viene definito un ciclo infinito che permette all'utente di inserire un prompt e ricevere una risposta dal modello e dal giudice, inoltre ogni 5 iterazioni viene fatto un riassunto della history.
"""


from assistant import Assistant

if __name__ == "__main__":
    
    assistant = Assistant()
    history_summary = ""
    i = 0
    while True:
        if i > 4:
            print("La history è troppo lunga: riassunto in corso!")
            history_summary = assistant.get_history_summary()
            i = 0
        user_input = input("Inserisci il prompt o digita 'exit' per terminare il programma: ")
        if user_input.lower() == "exit":
            print("Programma terminato!")
            break
        risposta, risposta_giudice = assistant.Ask(user_input, history_summary)
        print(f"Risposta llm:\n{risposta}\n")
        print(f"Risposta giudice:\n{risposta_giudice}\n")
        i = i+1