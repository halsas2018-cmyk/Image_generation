
import os
import time
import torch
from diffusers import StableDiffusionPipeline

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LORA_MODEL = "whitebearhands/lineart-lora"
OUTPUT_FILE = "sketch.png"

PROMPT = """
lineart_style, a simple educational whiteboard drawing of one apple,
one short stem and one leaf, bold clean black marker outlines,
clear continuous outer contour, centered isolated object,
pure white background, high contrast, simple hand-drawn illustration,
large open shapes, minimal interior details, no shading,
no gradients, no texture, no scenery
"""

NEGATIVE_PROMPT = """
photorealistic, 3D, painting, color gradients, gray background,
shadows, crosshatching, pencil shading, messy sketch, blurry lines,
excessive detail, multiple apples, extra leaves, text, watermark,
cropped object, colored background
"""


def main():
    torch.set_num_threads(2)
    started = time.time()

    print("Loading Stable Diffusion 1.5 on CPU...")

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    pipe.load_lora_weights(LORA_MODEL)
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()

    generator = torch.Generator(device="cpu").manual_seed(42)

    print("Generating test image...")

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=512,
        height=512,
        num_inference_steps=12,
        guidance_scale=6.5,
        generator=generator,
    )

    image = result.images[0]
    image.save(OUTPUT_FILE)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Image size: {image.size}")
    print(f"Elapsed time: {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
  
