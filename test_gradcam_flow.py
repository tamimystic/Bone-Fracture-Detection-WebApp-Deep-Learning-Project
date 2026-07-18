import sys
import os
import torch
import numpy as np
from PIL import Image

# Add the project directory to path
sys.path.append(os.path.abspath("."))

from model import get_model
from inference import predict_with_outputs
from gradcam import save_gradcam
from config import GRADCAM_FOLDER

def test():
    try:
        model = get_model()
        print("Model loaded.")
        
        # Create a dummy image
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        
        result = predict_with_outputs(img)
        print("Prediction done.")
        
        gradcam_path = os.path.join(GRADCAM_FOLDER, "test_gradcam.png")
        save_gradcam(
            image=img,
            model=model,
            tensor=result["tensor"],
            outputs=result["outputs"],
            class_idx=result["class_index"],
            save_path=gradcam_path,
        )
        print("Gradcam done.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
