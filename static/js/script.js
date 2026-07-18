const form = document.getElementById("uploadForm"),
imageInput = document.getElementById("image"),
browseBtn = document.getElementById("browseBtn"),
resetBtn = document.getElementById("resetBtn"),
dropArea = document.getElementById("dropArea"),
preview = document.getElementById("preview"),
previewPlaceholder = document.getElementById("previewPlaceholder"),
loading = document.getElementById("loading"),
resultSection = document.getElementById("resultSection"),
prediction = document.getElementById("prediction"),
confidence = document.getElementById("confidence"),
time = document.getElementById("time"),
probabilityTable = document.getElementById("probabilityTable"),
originalImage = document.getElementById("originalImage"),
processedImage = document.getElementById("processedImage"),
gradcamImage = document.getElementById("gradcamImage"),
widthField = document.getElementById("width"),
heightField = document.getElementById("height"),
formatField = document.getElementById("format"),
modeField = document.getElementById("mode"),
sizeField = document.getElementById("size");

browseBtn.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) previewImage(file);
});

["dragenter","dragover"].forEach(event => {
    dropArea.addEventListener(event, e => {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.add("dragging");
    });
});

["dragleave","dragend","drop"].forEach(event => {
    dropArea.addEventListener(event, e => {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.remove("dragging");
    });
});

dropArea.addEventListener("drop", e => {
    const files = e.dataTransfer.files;

    if (!files || !files.length) return;

    imageInput.files = files;
    previewImage(files[0]);
});

function previewImage(file){

    if(!(file instanceof File)){
        console.error("Invalid file.");
        return;
    }

    if(!file.type.startsWith("image/")){
        alert("Please select a valid image.");
        return;
    }

    const reader = new FileReader();

    reader.onload = e => {

        preview.onload = () => {
            preview.style.display = "block";
            preview.style.opacity = "1";
            previewPlaceholder.style.display = "none";
        };

        preview.onerror = () => {
            console.error("Preview image failed to load.");
            preview.removeAttribute("src");
            preview.style.display = "none";
            preview.style.opacity = "0";
            previewPlaceholder.style.display = "flex";
        };

        preview.src = e.target.result;
    };

    reader.onerror = () => {
        console.error("FileReader error.");
        alert("Unable to read the selected image.");
    };

    reader.readAsDataURL(file);
}

function toggleLoading(show){
    loading.classList.toggle("hidden", !show);
}

function toggleResult(show){
    resultSection.classList.toggle("hidden", !show);
}

function resetTable(){
    probabilityTable.innerHTML = "";
}

function updateProbabilityTable(probabilities){

    resetTable();

    Object.entries(probabilities).forEach(([label,value]) => {

        probabilityTable.insertAdjacentHTML(
            "beforeend",
            `
            <tr>
                <td>${label}</td>
                <td>${value}%</td>
            </tr>
            `
        );

    });

}

function updateImages(images){

    const imageList = [
        {element: originalImage, src: images.original},
        {element: processedImage, src: images.processed},
        {element: gradcamImage, src: images.gradcam}
    ];

    imageList.forEach(({element, src}) => {

        element.style.opacity = "0";

        element.onload = () => {
            element.style.opacity = "1";
        };

        element.onerror = () => {
            console.error("Failed to load:", src);
            element.removeAttribute("src");
            element.style.opacity = "0";
        };

        element.src = src;

    });

}

function updateInfo(info){

    widthField.textContent = info.width ?? "-";
    heightField.textContent = info.height ?? "-";
    formatField.textContent = info.format ?? "-";
    modeField.textContent = info.mode ?? "-";
    sizeField.textContent = info.size ? `${info.size} KB` : "-";

}

function resetUI(){

    form.reset();

    preview.removeAttribute("src");
    preview.style.display = "none";
    preview.style.opacity = "0";

    previewPlaceholder.style.display = "flex";

    prediction.textContent = "-";
    confidence.textContent = "-";
    time.textContent = "-";

    [originalImage, processedImage, gradcamImage].forEach(img => {
        img.removeAttribute("src");
        img.style.opacity = "0";
    });

    widthField.textContent = "-";
    heightField.textContent = "-";
    formatField.textContent = "-";
    modeField.textContent = "-";
    sizeField.textContent = "-";

    resetTable();

    toggleLoading(false);
    toggleResult(false);

}

resetBtn.addEventListener("click", resetUI);

form.addEventListener("submit", async e => {

    e.preventDefault();

    if(!imageInput.files.length){
        alert("Please select an image first.");
        return;
    }

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    toggleResult(false);
    toggleLoading(true);

    try{

        const response = await fetch("/predict",{
            method:"POST",
            body:formData
        });

        const data = await response.json();

        toggleLoading(false);

        if(!response.ok || !data.success){
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
            behavior:"smooth",
            block:"start"
        });

    }catch(err){

        console.error(err);

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