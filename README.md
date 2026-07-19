# Bone Fracture Detection Web Application

A deep learning-powered medical imaging web application designed to detect bone fractures from X-ray images. This project utilizes an EfficientNet-B0 model trained on medical data and features Gradient-weighted Class Activation Mapping (Grad-CAM) to provide transparent, explainable AI predictions.

---

## Overview

Medical image analysis requires both high accuracy and interpretability. This application provides a seamless web interface where users can upload X-ray images and receive instantaneous predictions regarding the presence of a fracture. To ensure the model's decisions are transparent, the application generates a Grad-CAM heatmap, highlighting the specific regions of the X-ray that heavily influenced the neural network's final classification.

The application is built with a focus on memory efficiency, allowing it to be deployed on resource-constrained cloud environments without triggering out-of-memory exceptions.

---

## Project Structure

Below is the complete directory structure of the repository:

```text
.
├── .git/                      # Git repository tracking
├── .vscode/                   # Editor-specific configurations
├── models/                    # Saved model weights
│   └── bone_fracture_model_final.pth
├── notebook/                  # Jupyter notebooks for model training and evaluation
│   ├── bone-fracture-detection-model_efficientnet_b0.ipynb
│   ├── bone-fracture-detection-model_resnet50.ipynb
│   └── *.png                  # Training metrics and confusion matrices
├── static/                    # Static assets for the Flask application
│   ├── css/                   # Stylesheets
│   ├── js/                    # Client-side JavaScript
│   ├── gradcam/               # Dynamically generated Grad-CAM heatmaps
│   ├── processed/             # Dynamically generated processed images
│   └── uploads/               # Temporary storage for user-uploaded X-rays
├── templates/                 # HTML templates
│   └── index.html             # Main frontend interface
├── .gitignore                 # Files and directories ignored by Git
├── .python-version            # Specified Python version for the environment
├── app.py                     # Main Flask application and API routing
├── config.py                  # Global configurations, paths, and hyperparameters
├── gradcam.py                 # Core Grad-CAM algorithm implementation
├── inference.py               # Optimized PyTorch inference pipeline
├── model.py                   # EfficientNet-B0 architecture definition
├── Procfile                   # Gunicorn configuration for production deployment
├── render.yaml                # Infrastructure-as-code for Render deployment
├── requirements.txt           # Python dependencies
├── runtime.txt                # Runtime specification for cloud platforms
├── template.py                # Automated script for generating project structure
├── test_app.py                # Local testing and debugging scripts
├── test_error.py              # Error simulation scripts
├── test_gradcam_flow.py       # Unit testing for the Grad-CAM pipeline
└── utils.py                   # Image preprocessing and file management utilities
```

---

## User Guide

The application is designed to be intuitive for both medical professionals and general users. 

### Usage Instructions
1. **Access the Application**: Navigate to the live deployment URL.
2. **Upload an X-ray**: Use the file selection button or the drag-and-drop area to upload an X-ray image. Supported formats include JPG, JPEG, PNG, BMP, TIF, TIFF, and WEBP.
3. **Execute Prediction**: Click the "Predict" button to send the image to the backend for analysis.
4. **Review Results**: The system will return a comprehensive report containing:
   - **Classification Result**: Indicates whether the X-ray is "Fractured" or "Normal".
   - **Confidence Score**: The statistical probability of the prediction.
   - **Explainable AI (Grad-CAM)**: A side-by-side comparison of the original X-ray and a generated heatmap. The heatmap highlights the exact focal points the AI analyzed to reach its conclusion.
   - **Metadata**: Technical details regarding the uploaded image file.

*Disclaimer: This application is a strictly educational and experimental tool. It should not be used as a substitute for professional medical diagnosis.*

---

## Developer Guide

The architecture separates the frontend presentation layer from the deep learning backend, communicating via asynchronous JSON responses.

### Technical Stack
- **Backend**: Python, Flask, Gunicorn
- **Machine Learning**: PyTorch, TorchVision
- **Base Architecture**: EfficientNet-B0 (Pre-trained on ImageNet, fine-tuned on X-ray data)
- **Computer Vision**: OpenCV, Pillow, NumPy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

### System Architecture and Memory Management
To facilitate deployment on free-tier cloud hosting (which typically imposes strict 512MB RAM limits), the backend architecture implements aggressive memory optimizations:
- **Inference Mode**: The initial prediction phase operates entirely within a `@torch.no_grad()` context, ensuring that no computational graphs are needlessly retained in memory.
- **Isolated Grad-CAM Execution**: The Grad-CAM pipeline independently recalculates the required forward and backward passes with `requires_grad=True`.
- **Garbage Collection**: Intermediate tensors and model outputs are forcefully deleted upon computation completion, followed by manual invocations of Python's garbage collector to prevent memory leaks across HTTP requests.

### Local Development Setup

Ensure Python 3.11 or higher is installed on your local machine.

1. **Clone the Repository**
   ```bash
   git clone https://github.com/tamimystic/Bone-Fracture-Detection-WebApp.git
   cd Bone-Fracture-Detection-WebApp
   ```

2. **Initialize a Virtual Environment**
   ```bash
   python -m venv .venv
   
   # Windows Activation
   .venv\Scripts\activate
   
   # Unix/MacOS Activation
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Model Assets**
   Confirm that the trained model file (`bone_fracture_model_final.pth`) is located within the `models/` directory. If it is missing, you may need to run the training notebooks provided in the `notebook/` directory.

5. **Launch the Server**
   ```bash
   python app.py
   ```
   The application will bind to `http://127.0.0.1:5000/`.

### Deployment

The project is fully configured for continuous deployment on cloud platforms.
- A `Procfile` is included to run the application using the Gunicorn WSGI HTTP Server.
- A `render.yaml` file is provided for automated infrastructure provisioning on Render.
- Memory handling is already optimized for standard containerized environments.

---
*Developed by [tamimystic](https://www.linkedin.com/in/tamimystic/)*
