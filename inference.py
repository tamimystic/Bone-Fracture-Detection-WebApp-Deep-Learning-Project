import time
import torch
import torch.nn.functional as F
from config import DEVICE, CLASS_NAMES
from model import load_model
from utils import preprocess_image, get_prediction, format_probabilities, inference_time

model = load_model()

@torch.inference_mode()
def predict(image):
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
        "time": inference_time(start)
    }

@torch.inference_mode()
def predict_tensor(tensor):
    outputs = model(tensor.to(DEVICE))
    probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
    return {
        "outputs": outputs,
        "class_index": int(probs.argmax()),
        "confidence": float(probs.max()),
        "probabilities": probs
    }

@torch.inference_mode()
def predict_from_path(path):
    from PIL import Image
    return predict(Image.open(path).convert("RGB"))

def predict_with_outputs(image):
    start = time.perf_counter()
    tensor = preprocess_image(image).to(DEVICE)

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
        "time": inference_time(start)
    }

def get_model():
    return model

def warmup():
    with torch.inference_mode():
        model(torch.randn(1, 3, 224, 224, device=DEVICE))