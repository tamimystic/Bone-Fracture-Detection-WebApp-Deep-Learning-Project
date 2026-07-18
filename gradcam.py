import gc
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

        self.forward_handle = layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, outputs, class_idx):
        self.model.zero_grad(set_to_none=True)

        outputs[:, class_idx].backward(retain_graph=False)

        if self.activations is None:
            raise RuntimeError("Forward hook failed.")

        if self.gradients is None:
            raise RuntimeError("Backward hook failed.")

        grads = self.gradients
        acts = self.activations

        grad2 = grads.pow(2)
        grad3 = grad2 * grads

        denom = 2 * grad2 + (acts * grad3).sum(
            dim=(2, 3),
            keepdim=True
        )

        denom = torch.where(
            denom != 0,
            denom,
            torch.ones_like(denom)
        )

        alpha = grad2 / (denom + 1e-7)

        weights = (
            alpha * torch.relu(grads)
        ).sum(
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.relu(
            (weights * acts).sum(dim=1)
        )

        cam = cam.squeeze()

        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:
            cam /= max_value

        cam = cam.detach().cpu().numpy()

        del grad2
        del grad3
        del denom
        del alpha
        del weights
        del grads
        del acts

        self.activations = None
        self.gradients = None

        return cam

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

        self.activations = None
        self.gradients = None


def get_heatmap(model, image, class_idx):
    gradcam = GradCAMPlusPlus(model)

    try:
        from utils import preprocess_image
        from config import DEVICE
        
        tensor = preprocess_image(image).to(DEVICE)
        tensor.requires_grad_(True)
        
        outputs = model(tensor)
        
        cam = gradcam.generate(outputs, class_idx)
        
        del tensor
        del outputs
        return cam

    finally:
        gradcam.close()

        del gradcam

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def overlay_heatmap(image, cam, alpha=GRADCAM_ALPHA):
    image = np.asarray(image)

    h, w = image.shape[:2]

    cam = cv2.resize(cam, (w, h))
    cam = np.uint8(cam * 255)

    heatmap = cv2.applyColorMap(
        cam,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    blended = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0,
    )

    return Image.fromarray(blended)


def get_gradcam_image(image, model, class_idx):
    cam = get_heatmap(
        model,
        image,
        class_idx,
    )

    return overlay_heatmap(
        image,
        cam,
    )


def save_gradcam(image, model, class_idx, save_path):
    result = get_gradcam_image(
        image,
        model,
        class_idx,
    )

    result.save(save_path)

    return save_path