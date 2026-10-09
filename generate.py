import os
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


MODEL_ID = "stabilityai/sd-turbo"
#MODEL_ID = "segmind/tiny-sd"
OUTPUT_DIR = Path("output")
OUTPUT_FILE = OUTPUT_DIR / "tiny_sd_test.png"

prompt = os.environ.get(
    "IMAGE_PROMPT",
    "A historically inspired illustration of an ancient African trading city, "
    "mud-brick architecture, traditional clothing, detailed hand-painted style"
)

# Use the available CPU threads without requiring a GPU.
torch.set_num_threads(min(4, torch.get_num_threads()))

print(f"Loading model: {MODEL_ID}")
print("Device: CPU")

start = time.perf_counter()

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)
pipe = pipe.to("cpu")

load_time = time.perf_counter() - start
print(f"Model loading time: {load_time:.1f} seconds")

print("Generating one 512x896 image...")

start = time.perf_counter()

image = pipe(
    prompt=prompt,
    width=512,
    height=896,
    num_inference_steps=4,
    guidance_scale=0.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

generation_time = time.perf_counter() - start

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT_FILE)

print(f"Generation time: {generation_time:.1f} seconds")
print(f"Image saved to: {OUTPUT_FILE}")
print(f"Image dimensions: {image.width}x{image.height}")
