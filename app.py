import gc
import os
from flask import Flask, render_template, request, jsonify

from config import (
    APP_NAME,
    SECRET_KEY,
    MAX_CONTENT_LENGTH,
    HOST,
    PORT,
    DEBUG,
    GRADCAM_FOLDER,
)
from utils import (
    allowed_file,
    save_upload,
    load_image,
    save_processed_image,
    image_info,
    clear_old_outputs,
)
from inference import predict_with_outputs
from model import get_model
from gradcam import save_gradcam

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

def static_url(folder, filename):
    return f"/static/{folder}/{filename}"

@app.route("/")
def index():
    return render_template("index.html", app_name=APP_NAME)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image uploaded."}), 400

    file = request.files["image"]

    if not file or file.filename == "":
        return jsonify({"success": False, "message": "Please select an image."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Unsupported image format."}), 400

    image = None

    try:
        clear_old_outputs()

        filename, image_path = save_upload(file)
        image = load_image(image_path)

        processed_name, _ = save_processed_image(image, filename)

        result = predict_with_outputs(image)
        model = get_model()

        gradcam_name = f"gradcam_{os.path.splitext(filename)[0]}.png"
        gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_name)

        save_gradcam(
            image=image,
            model=model,
            tensor=result["tensor"],
            outputs=result["outputs"],
            class_idx=result["class_index"],
            save_path=gradcam_path,
        )

        del result["tensor"]
        del result["outputs"]

        gc.collect()

        if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "time": result["time"],
            "image_info": image_info(image_path),
            "images": {
                "original": static_url("uploads", filename),
                "processed": static_url("processed", processed_name),
                "gradcam": static_url("gradcam", gradcam_name),
            },
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

    finally:
        if image is not None:
            image.close()
        gc.collect()

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "message": "Page not found."}), 404

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"success": False, "message": "Image size exceeds the allowed limit."}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "message": "Internal server error."}), 500

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)