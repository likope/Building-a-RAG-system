# import
import gradio as gr
from rag_core import Main

if __name__ == "__main__":
    # init the main
    main = Main()
    # define the gradio
    with gr.Blocks() as demo:
        inp = gr.Textbox(label="Prompt")
        send = gr.Button("Send")
        out_llm = gr.Textbox(label="LLM")
        out_judge = gr.Textbox(label="Judge")
        documents = gr.Textbox(label="Context")

        # on-click
        send.click(main.run_turn, inputs=inp, outputs=[out_llm, out_judge, documents])

    # launch the demo
    demo.launch(server_name="0.0.0.0")
