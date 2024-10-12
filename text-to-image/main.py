import gradio as gr
import torch
from diffusers import FluxPipeline
from transformers import MarianMTModel, MarianTokenizer


def fake_diffusion(prompt):
    src_text = [
        prompt,
    ]

    model_id = "G:\\opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id)
    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    res = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    print("翻译结果：", res)

    prompt = res[0]
    seed = 42
    model_id = "G:\\FLUX.1-schnell"

    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=4,
        generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]

    return image

if __name__ == "__main__":
    demo = gr.Interface(
        fake_diffusion,
        inputs=gr.Textbox(
            label="提示词",
            info="输入提示词，需等待一会才能生成图片",
            lines=3,
            value="大街上，走着一个性感火辣的年轻女子，她身着清凉，美丽而自信，在夜色的灯光下，格外诱人。",
        ),
        outputs="image")

    demo.launch(server_name="localhost", server_port=7860)