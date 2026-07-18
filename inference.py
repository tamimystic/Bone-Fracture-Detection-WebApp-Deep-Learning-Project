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

    probs = F.softmax(outputs, dim=1).squeeze()
    probs_np = probs.cpu().numpy()

    pred = get_prediction(probs_np, CLASS_NAMES)

    del tensor, outputs, probs

    return {
        "prediction": pred["label"],
        "class_index": pred["index"],
        "confidence": pred["confidence"],
        "probabilities": format_probabilities(probs_np, CLASS_NAMES),
        "time": inference_time(start),
    }


@torch.inference_mode()
def predict_tensor(tensor):
    model = get_model()

    outputs = model(tensor.to(DEVICE))

    probs = F.softmax(outputs, dim=1).squeeze()
    probs_np = probs.cpu().numpy()

    result = {
        "outputs": outputs,
        "class_index": int(probs_np.argmax()),
        "confidence": float(probs_np.max()),
        "probabilities": probs_np,
    }

    del probs

    return result


@torch.inference_mode()
def predict_from_path(path):
    with Image.open(path) as image:
        return predict(image.convert("RGB"))


def predict_with_outputs(image):
    model = get_model()

    start = time.perf_counter()

    tensor = preprocess_image(image).to(DEVICE)
    tensor.requires_grad_(True)

    outputs = model(tensor)

    probs = F.softmax(outputs, dim=1).squeeze()
    probs_np = probs.detach().cpu().numpy()

    pred = get_prediction(probs_np, CLASS_NAMES)

    del probs

    return {
        "tensor": tensor,
        "outputs": outputs,
        "prediction": pred["label"],
        "class_index": pred["index"],
        "confidence": pred["confidence"],
        "probabilities": format_probabilities(probs_np, CLASS_NAMES),
        "raw_probabilities": probs_np,
        "time": inference_time(start),
    }