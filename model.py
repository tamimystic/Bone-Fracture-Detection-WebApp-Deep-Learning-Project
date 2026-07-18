import torch
import torch.nn as nn
from torchvision.models import resnet50
from config import MODEL_PATH, DEVICE, NUM_CLASSES

def build_model():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def load_model():
    model = build_model()

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True if torch.__version__ >= "2.0" else False
    )

    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    cleaned = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(cleaned, strict=True)
    model.to(DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model

def get_target_layer(model):
    return model.layer4[-1]