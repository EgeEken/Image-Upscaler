
print("---\n\ninstalling necessary models and packages\n\n---")


import time
start = time.time()

from transformers import pipeline
import torch

import os


device = "cuda" if torch.cuda.is_available() else "cpu"
models = {
    "CAIDAS | Lightweight x2-64": "caidas/swin2SR-lightweight-x2-64",
    "CAIDAS | Realworld SR x4-64": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    "CAIDAS | Classical SR x2-64": "caidas/swin2SR-classical-sr-x2-64",
}


def load_model(model_id=None):
    global pipe
    if model_id is not None:
        pipe = pipeline(task="image-to-image", model=models[model_id], device=device)
    else:
        pipe = pipeline(task="image-to-image", model="caidas/swin2SR-lightweight-x2-64", device=device)

for model_id in models.keys():
    pipe = load_model(model_id)
    
print(f"---\n\nTime to load models: {round(time.time() - start, 2)} seconds\n\n---")