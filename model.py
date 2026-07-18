import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from config import MODEL_PATH, DEVICE, NUM_CLASSES

def build_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def load_model():
    model = build_model()

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True if torch.__version__ >= "2.0" else False
    )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "")] = v

    model.load_state_dict(cleaned)
    model.to(DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model

def get_target_layer(model):
    return model.layer4[-1]