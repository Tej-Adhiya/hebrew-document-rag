import gradio as gr
from process_user_query import process_query

# Create Gradio interface with new layout
with gr.Blocks(title="Hebrew Document Retrieval System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Hebrew Document Retrieval System")
    
    with gr.Row():
        # First Column: Query Input
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Enter your query in Hebrew",
                placeholder="Type your query here...",
                lines=4
            )
            submit_btn = gr.Button("Search", variant="primary")
        
        # Second Column: AI Response
        with gr.Column(scale=2):
           
            ai_output_box = gr.Textbox(
                label="AI Response",
                lines=10,
                interactive=False
            )
            response_output = ai_output_box
    
    # Second Row: Retrieved Documents
    with gr.Row():
        output_table = gr.Dataframe(
            headers=["Content", "Score", "Metadata"],
            label="Retrieved Documents",
            wrap=True,
            datatype=["str", "str", "json"]
        )
    
    submit_btn.click(
        fn=process_query,
        inputs=text_input,
        outputs=[output_table, response_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")