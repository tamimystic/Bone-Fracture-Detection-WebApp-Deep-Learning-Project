import numpy as np
import torch
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import transforms
from config import DEVICE, IMAGE_SIZE, RESIZE_SIZE, MEAN, STD, LIME_NUM_SAMPLES, LIME_NUM_FEATURES, LIME_POSITIVE_ONLY, LIME_HIDE_REST, LIME_MIN_WEIGHT

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

class LimeExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()

    def _predict(self, images):
        batch = torch.stack([transform(img.astype(np.uint8)) for img in images]).to(DEVICE)
        with torch.no_grad():
            return torch.softmax(self.model(batch), dim=1).cpu().numpy()

    def explain(self, image, class_idx):
        image = np.array(image)
        explanation = self.explainer.explain_instance(
            image,
            self._predict,
            top_labels=2,
            hide_color=0,
            num_samples=LIME_NUM_SAMPLES
        )

        result, mask = explanation.get_image_and_mask(
            class_idx,
            positive_only=LIME_POSITIVE_ONLY,
            hide_rest=LIME_HIDE_REST,
            num_features=LIME_NUM_FEATURES,
            min_weight=LIME_MIN_WEIGHT
        )

        result = mark_boundaries(result / 255.0, mask, color=(0, 1, 0), outline_color=(1, 1, 0), mode="thick")
        return Image.fromarray((result * 255).astype(np.uint8))

lime = None

def get_lime(model):
    global lime
    if lime is None:
        lime = LimeExplainer(model)
    return lime

def generate_lime(image, model, class_idx, save_path):
    result = get_lime(model).explain(image, class_idx)
    result.save(save_path)
    return save_path

def get_lime_image(image, model, class_idx):
    return get_lime(model).explain(image, class_idx)