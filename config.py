import os
import secrets
import torch

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "bone_fracture_model_final.pth")

STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(STATIC_DIR, "processed")
GRADCAM_FOLDER = os.path.join(STATIC_DIR, "gradcam")
LIME_FOLDER = os.path.join(STATIC_DIR, "lime")

for folder in (
    MODEL_DIR,
    STATIC_DIR,
    UPLOAD_FOLDER,
    PROCESSED_FOLDER,
    GRADCAM_FOLDER,
    LIME_FOLDER,
    TEMPLATE_DIR,
):
    os.makedirs(folder, exist_ok=True)

APP_NAME = "Bone Fracture Detection"
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
RESIZE_SIZE = 256

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

NUM_CLASSES = 2

CLASS_TO_IDX = {
    "fracture": 0,
    "normal": 1,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
CLASS_NAMES = list(CLASS_TO_IDX.keys())

MODEL_NAME = "ResNet50"
TARGET_LAYER = "layer4"

MAX_CONTENT_LENGTH = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "tif",
    "tiff",
    "webp",
}

SUPPORTED_FORMATS = tuple(ALLOWED_EXTENSIONS)

CONFIDENCE_DECIMALS = 2

GRADCAM_ALPHA = 0.45

LIME_NUM_SAMPLES = 1000
LIME_NUM_FEATURES = 8
LIME_POSITIVE_ONLY = True
LIME_HIDE_REST = False
LIME_MIN_WEIGHT = 0.01

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

TORCH_NUM_THREADS = max(os.cpu_count() // 2, 1)
torch.set_num_threads(TORCH_NUM_THREADS)

PREDICTION_LABELS = {
    "fracture": "Fracture",
    "normal": "Normal",
}