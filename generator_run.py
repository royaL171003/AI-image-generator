import gradio as gr
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import requests
from PIL import Image
from io import BytesIO

from examples.inference.image_to_image import StableDiffusionImg2ImgPipeline, preprocess

lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    scheduler=lms,
    revision="fp16", 
    use_auth_token=True
).to("cuda")

pipeimg = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
).to("cuda")




block = gr.Blocks(css=".container { max-width: 800px; margin: auto; }")

num_samples = 2

def infer(prompt, init_image, strength):
    if init_image != None:
        init_image = init_image.resize((512, 512))
        init_image = preprocess(init_image)
        with autocast("cuda"):
            images = pipeimg([prompt] * num_samples, init_image=init_image, strength=strength, guidance_scale=7.5)["sample"]
    else: 
        with autocast("cuda"):
            images = pipe([prompt] * num_samples, guidance_scale=7.5)["sample"]

    return images


with block as demo:
    gr.Markdown("<h1><center>Stable Diffusion</center></h1>")
    gr.Markdown(
        "Stable Diffusion is an AI model that generates images from any prompt you give!"
    )
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):

                text = gr.Textbox(
                    label="Enter your prompt", show_label=False, max_lines=1
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Run").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
        strength_slider = gr.Slider(
            label="Strength",
            maximum = 1,
            value = 0.75         
        )
        image = gr.Image(
            label="Intial Image",
            type="pil"
        )
               
        gallery = gr.Gallery(label="Generated images", show_label=False).style(
            grid=[2], height="auto"
        )
        text.submit(infer, inputs=[text,image,strength_slider], outputs=gallery)
        btn.click(infer, inputs=[text,image,strength_slider], outputs=gallery)

    gr.Markdown(
        """___
   <p style='text-align: center'>
   Created by CompVis and Stability AI
   <br/>
   </p>"""
    )


demo.launch(debug=True)