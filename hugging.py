import time
start = time.time()
from flask import Flask, render_template, request, send_from_directory

from transformers import pipeline
import torch
from PIL import Image

import os
print(f"---\n\nTime to load libraries: {round(time.time() - start, 2)} seconds\n\n---")

#try:
#    os.system("python imports.py")
#except:
#    print("Couldn't run imports.py, file might be missing or python might not be in path")

os.makedirs("outputs", exist_ok=True)


app = Flask(__name__) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"---\n\nDevice: {device}\n\n---")

models = {
    "CAIDAS | Lightweight x2-64": "caidas/swin2SR-lightweight-x2-64",
    "CAIDAS | Realworld SR x4-64": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    "CAIDAS | Classical SR x2-64": "caidas/swin2SR-classical-sr-x2-64",
}

model_id = "CAIDAS | Lightweight x2-64"

def load_model(model_id=None):
    global pipe
    if model_id is not None:
        pipe = pipeline(task="image-to-image", model=models[model_id], device=device)
    else:
        pipe = pipeline(task="image-to-image", model="caidas/swin2SR-lightweight-x2-64", device=device)

pipe = load_model(model_id)

@app.route('/')
def index():
    return render_template('index.html', available_models=models.keys())

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('index.html', error='No image file')

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error='No selected image file')

    old_file_path = os.path.join("outputs", "upscaled.png")
    if os.path.exists(old_file_path):
        os.remove(old_file_path)
        print("old file removed")
        
    upscaled_img = process_image(file)
    
    print("---\n\nUpscaled image saved as  upscaled.png \n\n---")
 
    return render_template('index.html', result=upscaled_img)

@app.route('/select_model', methods=['POST'])
def select_model():
    model_id = request.form.get('model_select')
    load_model(model_id)
    print(pipe.model.config)
    return render_template('index.html', available_models=models.keys())

@app.route('/outputs/<filename>')
def uploaded_file(filename):
    return send_from_directory("outputs", filename)

def process_image(file):
    print(f"---\n\nProcessing image {file.filename}\n\n---")
    
    img = Image.open(file)
    img = img.convert('RGB')

    start = time.time()

    result = pipe(img)
    
    print(f"---\n\nUpscaled image from size {img.size} to {result.size}\n\n---")
    print(f"---\n\nTime to process image: {round(time.time() - start, 2)} seconds\n\n---")
    upscaled_img = result.convert('RGB')
    upscaled_img.save(os.path.join("outputs", "upscaled.png"))
    return upscaled_img

if __name__ == '__main__':
    app.run(debug=True)