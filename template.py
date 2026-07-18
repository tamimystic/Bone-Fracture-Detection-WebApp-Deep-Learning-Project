from pathlib import Path

project = "Bone-Fracture-WebApp"

dirs = [
    "models",
    "static/css",
    "static/js",
    "static/uploads",
    "static/gradcam",
    "static/lime",
    "templates"
]

files = [
    "app.py",
    "requirements.txt",
    "Procfile",
    "runtime.txt",
    "render.yaml",
    "model.py",
    "inference.py",
    "gradcam.py",
    "lime_explainer.py",
    "utils.py",
    "config.py",
    "static/css/style.css",
    "static/js/script.js",
    "templates/index.html",
    "models/bone_fracture_model_final.pth"
]

root = Path(project)
root.mkdir(exist_ok=True)

for d in dirs:
    (root / d).mkdir(parents=True, exist_ok=True)

for f in files:
    path = root / f
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

print(f"\nProject '{project}' created successfully.\n")

for p in sorted(root.rglob("*")):
    indent = "    " * (len(p.relative_to(root).parts) - 1)
    icon = "📁" if p.is_dir() else "📄"
    print(f"{indent}{icon} {p.name}")