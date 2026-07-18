import time
import torch
import torch.nn.functional as F
from PIL import Image

from config import DEVICE, CLASS_NAMES
from model import get_model
from utils import (
    preprocess_image,
    get_prediction,
    format_probabilities,
    inference_time,
)

@torch.inference_mode()
def predict(image):
    model = get_model()

    start = time.perf_counter()

    tensor = preprocess_image(image).to(DEVICE)
    outputs = model(tensor)

    probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
    pred = get_prediction(probs, CLASS_NAMES)

    return {
        "prediction": pred["label"],
        "class_index": pred["index"],
        "confidence": pred["confidence"],
        "probabilities": format_probabilities(probs, CLASS_NAMES),
        "time": inference_time(start),
    }

@torch.inference_mode()
def predict_tensor(tensor):
    model = get_model()

    outputs = model(tensor.to(DEVICE))
    probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

    return {
        "outputs": outputs,
        "class_index": int(probs.argmax()),
        "confidence": float(probs.max()),
        "probabilities": probs,
    }

@torch.inference_mode()
def predict_from_path(path):
    with Image.open(path) as image:
        return predict(image.convert("RGB"))

def predict_with_outputs(image):
    model = get_model()

    start = time.perf_counter()

    tensor = preprocess_image(image).to(DEVICE)
    tensor.requires_grad_(True)

    with torch.enable_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze().detach().cpu().numpy()

    pred = get_prediction(probs, CLASS_NAMES)

    return {
        "tensor": tensor,
        "outputs": outputs,
        "prediction": pred["label"],
        "class_index": pred["index"],
        "confidence": pred["confidence"],
        "probabilities": format_probabilities(probs, CLASS_NAMES),
        "raw_probabilities": probs,
        "time": inference_time(start),
    }