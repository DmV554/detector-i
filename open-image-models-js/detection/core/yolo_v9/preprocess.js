/**
 * @fileoverview Preprocessing functions for YOLOv9 models using Canvas API.
 * Equivalent to open_image_models/detection/core/yolo_v9/preprocess.py
 */

/**
 * Creates a canvas element with the specified dimensions.
 * Helper function (can be moved to a utils file).
 * @param {number} width
 * @param {number} height
 * @returns {{canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D}}
 */
function createCanvas(width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        throw new Error("Could not get 2D context from canvas");
    }
    return { canvas, ctx };
}


/**
 * Resizes and pads an input image source to a target shape using the letterbox method.
 * Maintains aspect ratio and adds padding. Uses Canvas API.
 *
 * @param {HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|ImageBitmap} imageSource - The source image.
 * @param {number | {width: number, height: number}} newShape - The target shape (e.g., 640 or {width: 640, height: 640}).
 * @param {object} [options={}] - Optional parameters.
 * @param {{r: number, g: number, b: number}} [options.color={r: 114, g: 114, b: 114}] - RGB color for padding.
 * @param {boolean} [options.scaleup=true] - Allow scaling up if target shape is larger than source.
 * @returns {{canvas: HTMLCanvasElement, ratio: {w: number, h: number}, padding: {w: number, h: number}}} - Object containing the processed canvas, resize ratio, and padding amounts (half total padding).
 */
function letterbox(imageSource, newShape, options = {}) {
    const {
        color = { r: 114, g: 114, b: 114 }, // Default gray padding
        scaleup = true
    } = options;

    const shape = {
        width: imageSource.naturalWidth || imageSource.videoWidth || imageSource.width,
        height: imageSource.naturalHeight || imageSource.videoHeight || imageSource.height
    };

    let targetShape = {};
    if (typeof newShape === 'number') {
        targetShape = { width: newShape, height: newShape };
    } else if (typeof newShape === 'object' && newShape.width && newShape.height) {
        targetShape = { width: newShape.width, height: newShape.height };
    } else {
        throw new Error("Invalid newShape provided. Use number or {width, height}.");
    }

    // --- LOGS AÑADIDOS ---
    console.log(`DEBUG Letterbox: Input shape: ${shape.width}x${shape.height}`);
    console.log(`DEBUG Letterbox: Target shape: ${targetShape.width}x${targetShape.height}`);
    // --------------------

    // Calculate resize ratio (r = min(new / old))
    let r = Math.min(targetShape.height / shape.height, targetShape.width / shape.width);
    // --- LOG AÑADIDO ---
    console.log(`DEBUG Letterbox: Calculated initial ratio r: ${r}`);
    // --------------------

    if (!scaleup) {
        r = Math.min(r, 1.0); // Only scale down
        // --- LOG AÑADIDO ---
        console.log(`DEBUG Letterbox: Ratio after scaleup check: ${r}`);
        // --------------------
    }

    // Calculate new dimensions after resize (before padding)
    const newUnpad = {
        width: Math.round(shape.width * r),
        height: Math.round(shape.height * r)
    };

    // Calculate padding required (dw, dh are padding on each side)
    const dw = (targetShape.width - newUnpad.width) / 2;
    const dh = (targetShape.height - newUnpad.height) / 2;

    // --- LOGS AÑADIDOS ---
    console.log(`DEBUG Letterbox: Calculated newUnpad: ${newUnpad.width}x${newUnpad.height}`);
    console.log(`DEBUG Letterbox: Calculated padding dw: ${dw}, dh: ${dh}`);
    // --------------------

    // Create output canvas
    // ASUMO que tienes una función createCanvas definida en alguna parte
    const { canvas, ctx } = createCanvas(targetShape.width, targetShape.height);

    // --- Draw onto canvas ---
    // 1. Fill background with padding color
    ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
    ctx.fillRect(0, 0, targetShape.width, targetShape.height);

    // 2. Draw the resized image onto the canvas at the correct offset
    const drawX = dw;
    const drawY = dh;
    ctx.drawImage(imageSource, drawX, drawY, newUnpad.width, newUnpad.height);


    const result = {
        canvas: canvas,
        ratio: { w: r, h: r }, // Usa el valor final de 'r'
        padding: { w: dw, h: dh }
    };
    // --- LOG AÑADIDO ---
    console.log("DEBUG Letterbox: Returning -> ratio:", JSON.stringify(result.ratio), "padding:", JSON.stringify(result.padding));
    // --------------------
    return result;
}


/**
 * Preprocesses an image source for YOLOv9 model inference using the Canvas API.
 * Includes letterboxing, normalization (0-1), and conversion to NCHW format (RGB).
 *
 * @export
 * @param {HTMLImageElement|HTMLCanvasElement|HTMLVideoElement|ImageBitmap} imageSource - The source image.
 * @param {number | {width: number, height: number}} targetSize - The target input size for the model (e.g., 640 or {width: 640, height: 640}).
 * @returns {{tensor: Float32Array, ratio: {w: number, h: number}, padding: {w: number, h: number}}} - Object containing the preprocessed tensor (Float32Array NCHW), the resize ratio, and padding info.
 */
export function preprocessYOLOv9(imageSource, targetSize) {

    // 1. Letterbox the image (resize and pad)
    const { canvas, ratio, padding } = letterbox(imageSource, targetSize, {
        color: { r: 114, g: 114, b: 114 }, // Default YOLO padding color
        scaleup: true
    });
    const width = canvas.width;
    const height = canvas.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        throw new Error("Could not get 2D context from letterboxed canvas");
    }


    // 2. Get pixel data (RGBA)
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data; // Uint8ClampedArray [R, G, B, A, R, G, B, A, ...]

    // 3. Create output Float32Array in NCHW format (Batch=1, Channels=3)
    const tensorData = new Float32Array(1 * 3 * height * width);

    // 4. Normalize (0-255 -> 0-1) and transpose (RGBA HWC -> RGB CHW)
    //    Iterate through pixels and place them correctly in the NCHW tensor.
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pixelStartIndex = (y * width + x) * 4; // Index in RGBA array

            const r = data[pixelStartIndex] / 255.0;     // Normalize R
            const g = data[pixelStartIndex + 1] / 255.0; // Normalize G
            const b = data[pixelStartIndex + 2] / 255.0; // Normalize B
            // Alpha (data[pixelStartIndex + 3]) is ignored

            // Calculate indices in NCHW tensor
            const rIndex = 0 * height * width + y * width + x;
            const gIndex = 1 * height * width + y * width + x;
            const bIndex = 2 * height * width + y * width + x;

            tensorData[rIndex] = r;
            tensorData[gIndex] = g;
            tensorData[bIndex] = b;
        }
    }

    return {
        tensor: tensorData, // Float32Array NCHW format
        ratio: ratio,       // { w: r, h: r }
        padding: padding    // { w: dw, h: dh }
    };
}


// --- Example Usage (in an async context in a browser) ---
/*
async function runPreprocessing() {
    const imgElement = document.getElementById('myImageElement'); // Assume you have an <img> element
    if (!imgElement || !imgElement.complete || imgElement.naturalWidth === 0) {
        console.error("Image not loaded or invalid.");
        return;
    }

    const targetSize = 384; // Example target size for the model

    try {
        const { tensor, ratio, padding } = preprocessYOLOv9(imgElement, targetSize);

        console.log("Preprocessing complete.");
        console.log("Output Tensor Shape (conceptual):", [1, 3, targetSize, targetSize]);
        console.log("Output Tensor Length:", tensor.length);
        console.log("Resize Ratio:", ratio);
        console.log("Padding:", padding);

        // Now 'tensor' can be used as input for ONNX Runtime Web session
        // const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, targetSize, targetSize]) };
        // const results = await session.run(feeds);
        // ... pass ratio and padding to postprocessing ...

    } catch (error) {
        console.error("Preprocessing failed:", error);
    }
}

// Make sure image is loaded before calling
// const img = document.getElementById('myImageElement');
// if (img.complete) {
//     runPreprocessing();
// } else {
//     img.onload = runPreprocessing;
// }
*/