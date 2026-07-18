import cv2
import torch
import numpy as np
from PIL import Image
from config import DEVICE, GRADCAM_ALPHA
from model import get_target_layer

class GradCAMPlusPlus:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        layer = get_target_layer(model)
        layer.register_forward_hook(self._forward_hook)
        layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, tensor, outputs, class_idx):
        self.model.zero_grad(set_to_none=True)
        outputs[:, class_idx].backward()

        grads = self.gradients
        acts = self.activations

        grad2 = grads.pow(2)
        grad3 = grad2 * grads

        denom = 2 * grad2 + (acts * grad3).sum(dim=(2, 3), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))

        alpha = grad2 / (denom + 1e-7)
        weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam.detach().cpu().numpy()

gradcam = None

def get_gradcam(model):
    global gradcam
    if gradcam is None:
        gradcam = GradCAMPlusPlus(model)
    return gradcam

def overlay_heatmap(image, cam, alpha=GRADCAM_ALPHA):
    image = np.array(image)
    h, w = image.shape[:2]

    cam = cv2.resize(cam, (w, h))
    cam = np.uint8(cam * 255)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlay)

def save_gradcam(image, model, tensor, outputs, class_idx, save_path):
    cam = get_gradcam(model).generate(tensor.to(DEVICE), outputs, class_idx)
    result = overlay_heatmap(image, cam)
    result.save(save_path)
    return save_path

def get_gradcam_image(image, model, tensor, outputs, class_idx):
    cam = get_gradcam(model).generate(tensor.to(DEVICE), outputs, class_idx)
    return overlay_heatmap(image, cam)

def get_heatmap(model, tensor, outputs, class_idx):
    return get_gradcam(model).generate(tensor.to(DEVICE), outputs, class_idx)