import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from config import MODEL_PATH, NUM_CLASSES, DEVICE

_model = None

def get_model():
    global _model

    if _model is not None:
        return _model

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    _model = model
    return _model

def get_target_layer(model):
    return model.features[-1]