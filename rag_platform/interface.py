import gradio as gr
from .store import EnhancedDocumentStore

doc_store = EnhancedDocumentStore()

def create_interface():
    with gr.Blocks() as demo:
        pdf_input = gr.File(label="Upload PDF")
        output = gr.Textbox(label="Answer")
        def process(pdf_file):
            doc_store.process_pdf(pdf_file.name)
            return "PDF processed!"
        def ask(question):
            return doc_store.query(question)["answer"]
        gr.Button("Process PDF").click(process, inputs=[pdf_input], outputs=[output])
        gr.Textbox(label="Question").submit(ask, inputs=[output], outputs=[output])
    return demo