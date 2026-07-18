const form=document.getElementById("uploadForm"),
imageInput=document.getElementById("image"),
browseBtn=document.getElementById("browseBtn"),
resetBtn=document.getElementById("resetBtn"),
dropArea=document.getElementById("dropArea"),
preview=document.getElementById("preview"),
previewPlaceholder=document.getElementById("previewPlaceholder"),
loading=document.getElementById("loading"),
resultSection=document.getElementById("resultSection"),
prediction=document.getElementById("prediction"),
confidence=document.getElementById("confidence"),
time=document.getElementById("time"),
probabilityTable=document.getElementById("probabilityTable"),
originalImage=document.getElementById("originalImage"),
processedImage=document.getElementById("processedImage"),
gradcamImage=document.getElementById("gradcamImage"),
limeImage=document.getElementById("limeImage"),
widthField=document.getElementById("width"),
heightField=document.getElementById("height"),
formatField=document.getElementById("format"),
modeField=document.getElementById("mode"),
sizeField=document.getElementById("size");

browseBtn.onclick=()=>imageInput.click();

imageInput.onchange=e=>{
    if(e.target.files.length) previewImage(e.target.files[0]);
};

["dragenter","dragover"].forEach(evt=>{
    dropArea.addEventListener(evt,e=>{
        e.preventDefault();
        dropArea.classList.add("dragging");
    });
});

["dragleave","dragend"].forEach(evt=>{
    dropArea.addEventListener(evt,()=>{
        dropArea.classList.remove("dragging");
    });
});

dropArea.addEventListener("drop",e=>{
    e.preventDefault();
    dropArea.classList.remove("dragging");

    if(!e.dataTransfer.files.length) return;

    imageInput.files=e.dataTransfer.files;
    previewImage(e.dataTransfer.files[0]);
});

function previewImage(file){
    if(!file.type.startsWith("image/")) return;

    const reader=new FileReader();

    reader.onload=e=>{
        preview.src=e.target.result;
        preview.style.display="block";
        previewPlaceholder.style.display="none";
    };

    reader.readAsDataURL(file);
}

function toggleLoading(show){
    loading.classList.toggle("hidden",!show);
}

function toggleResult(show){
    resultSection.classList.toggle("hidden",!show);
}

function resetTable(){
    probabilityTable.innerHTML="";
}

function updateProbabilityTable(probabilities){
    resetTable();

    Object.entries(probabilities).forEach(([label,value])=>{
        probabilityTable.insertAdjacentHTML(
            "beforeend",
            `<tr><td>${label}</td><td>${value}%</td></tr>`
        );
    });
}

function updateImages(images){
    originalImage.src=images.original;
    processedImage.src=images.processed;
    gradcamImage.src=images.gradcam;
    limeImage.src=images.lime;
}

function updateInfo(info){
    widthField.textContent=info.width;
    heightField.textContent=info.height;
    formatField.textContent=info.format;
    modeField.textContent=info.mode;
    sizeField.textContent=info.size;
}

function resetUI(){
    form.reset();

    preview.removeAttribute("src");
    preview.style.display="none";

    previewPlaceholder.style.display="flex";

    prediction.textContent="-";
    confidence.textContent="-";
    time.textContent="-";

    originalImage.removeAttribute("src");
    processedImage.removeAttribute("src");
    gradcamImage.removeAttribute("src");
    limeImage.removeAttribute("src");

    widthField.textContent="-";
    heightField.textContent="-";
    formatField.textContent="-";
    modeField.textContent="-";
    sizeField.textContent="-";

    resetTable();

    toggleLoading(false);
    toggleResult(false);
}

resetBtn.addEventListener("click",resetUI);

form.addEventListener("submit", async e => {
    e.preventDefault();

    if (!imageInput.files.length) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    toggleResult(false);
    toggleLoading(true);

    try {
        const res = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        toggleLoading(false);

        if (!res.ok || !data.success) {
            throw new Error(data.message || "Prediction failed.");
        }

        prediction.textContent = data.prediction;
        confidence.textContent = `${data.confidence}%`;
        time.textContent = data.time;

        updateProbabilityTable(data.probabilities);
        updateImages(data.images);
        updateInfo(data.image_info);

        toggleResult(true);

        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch (err) {
        toggleLoading(false);
        toggleResult(false);
        alert(err.message || "Something went wrong.");
    }
});

window.addEventListener("load", () => {
    toggleLoading(false);
    toggleResult(false);
});

window.addEventListener("dragover", e => e.preventDefault());
window.addEventListener("drop", e => e.preventDefault());

["originalImage", "processedImage", "gradcamImage", "limeImage"].forEach(id => {
    const img = document.getElementById(id);

    img.addEventListener("load", () => {
        img.style.opacity = "1";
    });

    img.addEventListener("error", () => {
        img.removeAttribute("src");
    });
});