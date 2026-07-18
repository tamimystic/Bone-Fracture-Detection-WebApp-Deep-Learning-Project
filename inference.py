import time
import torch
import torch.nn.functional as F

from config import DEVICE, CLASS_NAMES
from model import load_model
from utils import (
    preprocess_image,
    get_prediction,
    format_probabilities,
    inference_time,
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


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
    from PIL import Image

    image = Image.open(path).convert("RGB")
    return predict(image)


def predict_with_outputs(image):
    model = get_model()

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
        "time": inference_time(start),
    }


def warmup():
    model = get_model()

    with torch.inference_mode():
        dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
        model(dummy)