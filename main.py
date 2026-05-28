from assistant import Assistente

if __name__ == "__main__":
    assistant = Assistente()
    while True:
        user_input = input("Inserisci il prompt: ")
        risposta = assistant.Ask(user_input)
        print(risposta)