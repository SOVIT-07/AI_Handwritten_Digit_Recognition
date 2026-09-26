const canvas =
    document.getElementById(
        "drawingCanvas"
    );

const ctx =
    canvas.getContext("2d");

const result =
    document.getElementById(
        "result"
    );

const imageInput =
    document.getElementById(
        "imageInput"
    );

const uploadButton =
    document.getElementById(
        "uploadButton"
    );

const predictDrawing =
    document.getElementById(
        "predictDrawing"
    );

const clearCanvas =
    document.getElementById(
        "clearCanvas"
    );


let drawing = false;


ctx.fillStyle = "black";

ctx.fillRect(
    0,
    0,
    canvas.width,
    canvas.height
);

ctx.strokeStyle = "white";

ctx.lineWidth = 18;

ctx.lineCap = "round";


canvas.addEventListener(
    "mousedown",
    (event) => {

        drawing = true;

        draw(event);
    }
);


canvas.addEventListener(
    "mouseup",
    () => {

        drawing = false;

        ctx.beginPath();
    }
);


canvas.addEventListener(
    "mouseleave",
    () => {

        drawing = false;

        ctx.beginPath();
    }
);


canvas.addEventListener(
    "mousemove",
    draw
);


function draw(event) {

    if (!drawing) {
        return;
    }

    const rect =
        canvas.getBoundingClientRect();

    const x =
        event.clientX -
        rect.left;

    const y =
        event.clientY -
        rect.top;

    ctx.lineTo(x, y);

    ctx.stroke();

    ctx.beginPath();

    ctx.moveTo(x, y);
}


clearCanvas.addEventListener(
    "click",
    () => {

        ctx.fillStyle = "black";

        ctx.fillRect(
            0,
            0,
            canvas.width,
            canvas.height
        );

        ctx.strokeStyle = "white";

        result.textContent =
            "Draw or upload a digit to begin.";
    }
);


predictDrawing.addEventListener(
    "click",
    async () => {

        canvas.toBlob(
            async (blob) => {

                const formData =
                    new FormData();

                formData.append(
                    "file",
                    blob,
                    "drawing.png"
                );

                await sendPrediction(
                    formData
                );

            },
            "image/png"
        );
    }
);


uploadButton.addEventListener(
    "click",
    async () => {

        const file =
            imageInput.files[0];

        if (!file) {

            result.textContent =
                "Please select an image first.";

            return;
        }

        const formData =
            new FormData();

        formData.append(
            "file",
            file
        );

        await sendPrediction(
            formData
        );
    }
);


async function sendPrediction(
    formData
) {

    result.textContent =
        "Predicting...";

    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        if (data.success) {

            result.innerHTML =
                `Predicted Digit:
                <strong>${data.digit}</strong>
                <br>
                Confidence:
                <strong>${data.confidence}%</strong>`;

        } else {

            result.textContent =
                "Error: " +
                data.error;
        }

    } catch (error) {

        result.textContent =
            "Prediction failed: " +
            error.message;
    }
}