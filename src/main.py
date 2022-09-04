from datetime import datetime as dt
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True)
pipe.save_config(".")
pipe = pipe.to(device)

prompt = "hyper realistic hardcore dungeon with black lights glowing purple dildos in photo style"
with autocast("cuda"):
    image = pipe(prompt, height=768, width=768, guidance_scale=7.5)["sample"][0]

stamp = int(dt.timestamp(dt.now()))
image.save(f"samples/testing-{stamp}.png")

barbras_cell_phone = 6025405175