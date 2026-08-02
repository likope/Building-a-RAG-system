import gradio as gr
from rag_core import Main

if __name__ == "__main__":
    main = Main()
    with gr.Blocks() as demo:
        inp       = gr.Textbox(label="Prompt")
        send      = gr.Button("Send")
        out_llm   = gr.Textbox(label="LLM")
        out_judge = gr.Textbox(label="Judge")
        documents = gr.Textbox(label="Context")

        reload_btn = gr.Button("Reload context")
        status     = gr.Textbox(label="Status")


        send.click(main.run_turn, inputs=inp, outputs=[out_llm, out_judge, documents])
        reload_btn.click(main.reload_context, inputs=None, outputs=status)

    demo.launch(server_name="0.0.0.0")