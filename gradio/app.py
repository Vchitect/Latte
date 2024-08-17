import os
import sys
from types import SimpleNamespace
from huggingface_hub import snapshot_download
import gradio as gr
import spaces

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from inference import inference_function, video_to_base64






@spaces.GPU(duration=200)
def run_inference(prompt_text, visual_prompt, is_running_in_api=None):
    model_id = "maxin-cn/Latte-1"

    # negative_prompt = "watermark+++, text, shutterstock text, shutterstock++, blurry, ugly, username, url, low resolution, low quality"
    negative_prompt = None
    args = {
        "model": model_id,
        "prompt": prompt_text,
        "negative_prompt": negative_prompt,
        "num_frames": 16,
        "num_steps": 50,
        # "width": 256,
        # "height": 256,
        # "visual_prompt": visual_prompt,
        "device": 'cuda',
        "quantize": True,
        "fps": 4,
        "output_dir": "./outputs",
    }

    print(is_running_in_api)
    responseFile = inference_function(SimpleNamespace(**args))
    print(model_id, "Produces -> ", responseFile)
    if is_running_in_api:
        base64_file = video_to_base64(src_path=responseFile, delete_src=False)
        return "data:video/mp4;base64,"+base64_file
    else:
        return responseFile


def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.HTML(
                """
                <br/>
                <h1 style='text-align: center'>
                    Latte: Efficient High Quality Video Production.
                </h1>
                <br/>
                """
            )

        with gr.Row():
            with gr.Column():
                prompt_text = gr.Textbox(show_label=False, placeholder="Enter prompt text here", lines=4)
                
                gr.Markdown("## Visual Prompt")
                visual_prompt = gr.Image(label="Image (optional)", show_download_button=True)
                submit_button = gr.Button("Run Inference")

            with gr.Column():
                output_video = gr.Video(label="Output Video", height="100%", autoplay=True, show_download_button=True)

        submit_button.click(
            fn=run_inference, 
            inputs=[
                prompt_text,
                visual_prompt,
            ], 
            outputs=output_video
        )
        gr.Examples(
            examples=[
                [ "A cat wearing sunglasses and working as a lifeguard at pool." ],
                [ "A car driving fast on the Eastern beach in East London." ]       
            ],
            fn=run_inference,
            inputs=[prompt_text,],
            outputs=[output_video],
            cache_examples=True,
        )


    # launch
    demo.queue(max_size=5, default_concurrency_limit=1)
    demo.launch(debug=True, share=False, max_threads=1)

if __name__ == "__main__":
    main()

# python gradio/app.py