import os, uuid, shutil
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from config import *

transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

inv_normalize = transforms.Normalize(mean=[-m / s for m, s in zip(MEAN, STD)], std=[1 / s for s in STD])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def create_filename(filename):
    ext = filename.rsplit(".", 1)[1].lower()
    return f"{uuid.uuid4().hex}.{ext}"

def save_upload(file):
    filename = create_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return filename, path

def load_image(path):
    return Image.open(path).convert("RGB")

def preprocess_image(image):
    return transform(image).unsqueeze(0)

def tensor_to_image(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    tensor = inv_normalize(tensor.detach().cpu()).clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    return image

def save_processed_image(image, filename):
    tensor = preprocess_image(image)
    processed = tensor_to_image(tensor)
    name = f"processed_{os.path.splitext(filename)[0]}.png"
    path = os.path.join(PROCESSED_FOLDER, name)
    processed.save(path)
    return name, path

def get_prediction(probabilities, class_names):
    idx = int(np.argmax(probabilities))
    return {"label": class_names[idx].title(), "index": idx, "confidence": round(float(probabilities[idx]) * 100, CONFIDENCE_DECIMALS)}

def format_probabilities(probabilities, class_names):
    return {class_names[i].title(): round(float(probabilities[i]) * 100, CONFIDENCE_DECIMALS) for i in range(len(class_names))}

def inference_time(start):
    import time
    return f"{(time.perf_counter() - start) * 1000:.2f} ms"

def image_info(path):
    image = Image.open(path)
    return {
        "width": image.width,
        "height": image.height,
        "format": image.format,
        "mode": image.mode,
        "size": round(os.path.getsize(path) / 1024, 2)
    }

def remove_file(path):
    if os.path.exists(path):
        os.remove(path)

def clear_directory(folder):
    if not os.path.exists(folder):
        return
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

def clear_old_outputs():
    for folder in (UPLOAD_FOLDER, PROCESSED_FOLDER, GRADCAM_FOLDER, LIME_FOLDER):
        clear_directory(folder)