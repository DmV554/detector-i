/**
 * @fileoverview YoloV9 Object Detector implementation using ONNX Runtime Web.
 * Equivalent structure to open_image_models/detection/core/yolo_v9/inference.py
 */

// Ensure ONNX Runtime is loaded (usually via CDN)
const ort = window.ort;
if (!ort) {
    throw new Error("ONNX Runtime Web (ort) not found. Ensure it's loaded, e.g., via CDN.");
}

import { ObjectDetector , DetectionResult } from '../base.js';
import { preprocessYOLOv9 } from './preprocess.js';
import { convertToDetectionResult } from './postprocess.js';

const LOGGER = {
    info: (...args) => console.log('[INFO]', ...args),
    warn: (...args) => console.warn('[WARN]', ...args),
    error: (...args) => console.error('[ERROR]', ...args),
};

/**
 * YoloV9-specific ONNX inference class for object detection in the browser.
 * @extends {ObjectDetector}
 */
export class YoloV9ObjectDetector extends ObjectDetector {
    /** @type {ort.InferenceSession | null} */
    session = null;
    /** @type {string[]} */
    classLabels = [];
    /** @type {number} */
    confThresh = 0.25;
    /** @type {string} */
    inputName = '';
    /** @type {string[]} */ // Models can have multiple outputs
    outputNames = [];
    /** @type {{width: number, height: number}} */
    imgSize = { width: 0, height: 0 };
    /** @type {string[]} */
    providers = [];
    /** @type {string} */
    modelName = "yolov9-model"; // Default or derived name
    /** @type {Promise<void> | null} */
    _initPromise = null;

    /**
     * Initializes the YoloV9ObjectDetector. Model loading is asynchronous.
     *
     * @param {ArrayBuffer|Uint8Array|Blob} modelData - The ONNX model data.
     * @param {string[]} classLabels - List of class labels corresponding to the class IDs.
     * @param {object} [options={}] - Configuration options.
     * @param {number} [options.confThresh=0.25] - Confidence threshold for filtering predictions.
     * @param {string[]} [options.providers=['wasm']] - Execution providers for ONNX Runtime (e.g., ['wasm', 'webgl']).
     * @param {ort.SessionOptions} [options.sessionOptions] - Advanced ONNX Runtime session options. Provider options should be set here.
     * @param {string} [options.modelName="yolov9-model"] - Optional name for the model (used in logging).
     */
    constructor(modelData, classLabels, options = {}) {
        super(); // Call base class constructor if needed

        this.classLabels = classLabels;
        this.confThresh = options.confThresh ?? 0.25;
        this.modelName = options.modelName ?? "yolov9-model";

        // Default session options focusing on browser providers
        const defaultProviders = ['wasm']; // 'webgl' is another common option
        this.providers = options.providers ?? defaultProviders;

        // Merge providers into sessionOptions if provided separately
        let sessionOptions = options.sessionOptions ?? {};
        if (!sessionOptions.executionProviders) {
             sessionOptions.executionProviders = this.providers;
        } else {
            // Log if providers differ between options.providers and options.sessionOptions
            if (options.providers && JSON.stringify(options.providers) !== JSON.stringify(sessionOptions.executionProviders)) {
                 LOGGER.warn(`Both options.providers and sessionOptions.executionProviders were specified, using sessionOptions: ${sessionOptions.executionProviders}`);
                 this.providers = sessionOptions.executionProviders;
            }
        }


        // Model initialization is async, start it here
        this._initPromise = this._initializeModel(modelData, sessionOptions);
        this._initPromise.catch(error => {
             LOGGER.error(`Failed to initialize model '${this.modelName}':`, error);
             // Prevent further operations if init fails
             this._initPromise = Promise.reject(error);
             this.session = null;
        });
    }

    /**
     * Asynchronously creates the ONNX InferenceSession.
     * @private
     * @param {ArrayBuffer|Uint8Array|Blob} modelData
     * @param {ort.SessionOptions} sessionOptions
     * @returns {Promise<void>}
     */
   async _initializeModel(modelData, sessionOptions) {
        LOGGER.info(`Initializing ONNX Runtime session for model '${this.modelName}'...`);
        LOGGER.info(`Using providers: ${sessionOptions.executionProviders}`);

        this.session = await ort.InferenceSession.create(modelData, sessionOptions);

        // --- DEBUG: Imprimir el objeto session y los nombres ---
        console.log('DEBUG: ORT Session Object:', this.session);
        // --- FIN DEBUG ---


        // Get input and output details
        this.inputName = this.session.inputNames[0];
        this.outputNames = this.session.outputNames; // Can be multiple

        // --- DEBUG: Imprimir nombres ---
        console.log('DEBUG: Input Name:', this.inputName);
        console.log('DEBUG: Output Names:', this.outputNames);
         // --- FIN DEBUG ---


        if (!this.inputName || this.outputNames.length === 0) {
             throw new Error("Failed to get input/output names from the model.");
        }

        // --- Workaround: Infer image size from model name ---
        // Extracts numbers like '608', '384' from names like 'yolo-v9-s-608-...'
        const sizeMatch = this.modelName.match(/-(\d+)-license-plate/);
        if (sizeMatch && sizeMatch[1]) {
            const size = parseInt(sizeMatch[1], 10);
            this.imgSize = { height: size, width: size };
            LOGGER.info(`Inferred input size from model name: ${size}x${size}`);
        } else {
            // Fallback if size not found in name - essential to set *something*
            const defaultSize = 384; // Choose a reasonable default or make it an option
            this.imgSize = { height: defaultSize, width: defaultSize };
            LOGGER.warn(`Could not infer input size from model name '${this.modelName}'. Falling back to default: ${defaultSize}x${defaultSize}.`);
            // Optionally, you could throw an error here if size is mandatory
            // throw new Error("Could not determine model input size.");
        }
        // --- END WORKAROUND ---

        LOGGER.info(`Model '${this.modelName}' initialized successfully (shape reading bypassed).`);
        LOGGER.info(`Using Input Name: ${this.inputName}, Expected Shape: [N, C, ${this.imgSize.height}, ${this.imgSize.width}] (Inferred)`);
        LOGGER.info(`Output Names: ${this.outputNames.join(', ')}`);
    }

    /**
     * Ensures the model session is initialized before proceeding.
     * @private
     * @returns {Promise<void>}
     */
    async _ensureInitialized() {
        if (!this._initPromise) {
            throw new Error("Model initialization was not started.");
        }
        // Wait for the initialization promise to resolve (or reject)
        await this._initPromise;
        if (!this.session) {
            // This case should be covered by the catch in the constructor, but double-check
            throw new Error(`Model session is not available for '${this.modelName}'. Initialization likely failed.`);
        }
    }

    /**
     * Perform object detection on a single image source or a list of sources.
     *
     * @override // Overrides method from ObjectDetector
     * @param {HTMLImageElement|HTMLCanvasElement|ImageBitmap|ImageData|HTMLVideoElement | (HTMLImageElement|HTMLCanvasElement|ImageBitmap|ImageData|HTMLVideoElement)[]} imageSources - A single image source or a list of image sources.
     * @returns {Promise<DetectionResult[] | DetectionResult[][]>} A list of detections for a single input, or a list of lists for multiple inputs.
     */
    async predict(imageSources) {
        await this._ensureInitialized();

        const runSingle = (source) => this._predict(source);

        if (Array.isArray(imageSources)) {
            // Process list of sources - currently sequentially, consider Promise.all for parallel *inference* if supported well
            // Note: Python version also processed sequentially due to batching limitations mentioned.
            const resultsList = [];
            for (const source of imageSources) {
                 resultsList.push(await runSingle(source));
            }
            return resultsList;
            // If parallel preprocessing & inference is desired (and potentially faster):
            // return Promise.all(imageSources.map(source => runSingle(source)));
        } else {
            // Process single source
            return runSingle(imageSources);
        }
    }

    /**
     * Perform object detection on a single image source.
     * @private
     * @param {HTMLImageElement|HTMLCanvasElement|ImageBitmap|ImageData|HTMLVideoElement} imageSource - Single image source.
     * @returns {Promise<DetectionResult[]>} List of detections.
     */
    async _predict(imageSource) {
        // 1. Preprocess the image
        // Ensure preprocessYOLOv9 uses the correct target size from the model
        const { tensor, ratio, padding } = preprocessYOLOv9(imageSource, this.imgSize);

        // 2. Create ONNX Tensor
        const inputTensor = new ort.Tensor('float32', tensor, [1, 3, this.imgSize.height, this.imgSize.width]);

        // 3. Run Inference
        let outputMap;
        try {
            // Feed format: { inputName: tensor }
            const feeds = { [this.inputName]: inputTensor };
            outputMap = await this.session.run(feeds);
        } catch (e) {
            LOGGER.error(`An error occurred during model inference: ${e.message}`);
            // Handle specific errors or return empty based on Python example
            // The CoreML specific error check from Python isn't directly applicable here.
             if (e.message.includes("CoreML")) { // Example check, might need refinement
                  LOGGER.warn("Inference potentially failed due to empty data scenario (CoreML issue?). Returning empty results.");
             }
            return []; // Return empty list on error
        }


                   // --- Añade estos logs ---
            console.log('DEBUG: Full Output Map from session.run:', outputMap);

            // Asumiendo que la salida principal es la primera
            const outputTensor = outputMap[this.outputNames[0]]; // outputNames[0] es 'output0'

            console.log(`DEBUG: Output Tensor ('${this.outputNames[0]}'):`, outputTensor);
            // --- Fin de los logs ---

            if (!outputTensor || !outputTensor.data || !outputTensor.dims) {
                LOGGER.warn("Model output tensor seems invalid or missing. Returning empty results.");
                return [];
            }

            console.log("DEBUG: Ratio pasado:", JSON.stringify(ratio));
            console.log("DEBUG: Padding pasado:", JSON.stringify(padding));
                // Llamada a postprocesamiento (donde ocurre el error)
        return convertToDetectionResult(
            outputTensor.data,
            outputTensor.dims, // <-- El valor [1, 7] que causa el error
            this.classLabels,
            ratio,
            padding,
            this.confThresh
        );
    }

    /**
     * Run and display benchmark results in the console.
     *
     * @override // Overrides method from ObjectDetector
     * @param {number} [numRuns=100] - Number of inference runs for averaging (reduced default for browser).
     */
    async showBenchmark(numRuns = 100) {
        await this._ensureInitialized();
        LOGGER.info(`Starting benchmark for model '${this.modelName}'...`);

        // Use a dummy input tensor for benchmarking
        const dummyInputTensor = new ort.Tensor(
             'float32',
             new Float32Array(1 * 3 * this.imgSize.height * this.imgSize.width), // Filled with zeros
             [1, 3, this.imgSize.height, this.imgSize.width]
         );
        const dummyFeeds = { [this.inputName]: dummyInputTensor };

        // Warm-up phase
        await this._warm_up(dummyFeeds, Math.min(numRuns, 50)); // Limit warmup runs

        // Measure performance
        const totalTimeMs = await this._benchmark_inference(dummyFeeds, numRuns);
        const avgTimeMs = totalTimeMs / numRuns;
        const fps = 1000 / avgTimeMs;

        // Display results
        this._display_benchmark_results(avgTimeMs, fps, numRuns);
    }

    /**
     * Warm-up phase for benchmarking.
     * @private
     */
    async _warm_up(feeds, numRuns = 50) {
        LOGGER.info(`Starting model warm-up with ${numRuns} runs...`);
        for (let i = 0; i < numRuns; i++) {
            try {
                await this.session.run(feeds);
            } catch (e) {
                 LOGGER.warn(`Warm-up run ${i+1} failed: ${e.message}`);
                 // Don't stop benchmark for warmup failures, but log them.
            }
        }
        LOGGER.info("Model warm-up completed.");
    }


    /**
     * Runs detection and draws results on a new canvas.
     *
     * @override // Overrides method from ObjectDetectorBase
     * @param {HTMLImageElement|HTMLCanvasElement|ImageBitmap|ImageData|HTMLVideoElement} imageSource - The source image.
     * @param {object} [options={}] - Drawing options.
     * @param {string} [options.lineColor='lime'] - Color for bounding boxes.
     * @param {number} [options.lineWidth=2] - Line width for bounding boxes.
     * @param {string} [options.textColor='black'] - Color for label text.
     * @param {string} [options.textBgColor='lime'] - Background color for label text.
     * @param {string} [options.font='14px sans-serif'] - Font for labels.
     * @returns {Promise<HTMLCanvasElement>} A new canvas element with the image and detections drawn.
     */
    async displayPredictions(imageSource, options = {}) {
        await this._ensureInitialized();

        const {
             lineColor = 'lime', // Brighter default than Python's green
             lineWidth = 2,
             textColor = 'black',
             textBgColor = 'lime',
             font = '14px sans-serif'
         } = options;


        // Get detections
        const detections = await this._predict(imageSource);

        // Create output canvas and draw the original image
        const originalWidth = imageSource.naturalWidth || imageSource.videoWidth || imageSource.width;
        const originalHeight = imageSource.naturalHeight || imageSource.videoHeight || imageSource.height;
        const outputCanvas = document.createElement('canvas');
        outputCanvas.width = originalWidth;
        outputCanvas.height = originalHeight;
        const ctx = outputCanvas.getContext('2d');
         if (!ctx) throw new Error("Could not get 2D context for output canvas");

        ctx.drawImage(imageSource, 0, 0, originalWidth, originalHeight);

        // Draw detections
        detections.forEach(detection => {
            const bb = detection.boundingBox;
            const label = `${detection.label}: ${detection.confidence.toFixed(2)}`;

            // Draw bounding box
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = lineWidth;
            ctx.strokeRect(bb.x1, bb.y1, bb.width, bb.height);

            // Prepare text properties
            ctx.font = font;
            ctx.textBaseline = 'bottom'; // Align text nicely above the box
            const textMetrics = ctx.measureText(label);
            const textWidth = textMetrics.width;
            // Estimate text height based on font size (crude but often works)
            const textHeight = parseInt(font, 10) * 1.2;

            const textX = bb.x1 + lineWidth / 2; // Align with box edge
            const textY = bb.y1 - lineWidth / 2; // Position above the box

            // Draw text background rectangle
            ctx.fillStyle = textBgColor;
            ctx.fillRect(textX, textY - textHeight, textWidth, textHeight);

            // Draw text
            ctx.fillStyle = textColor;
            ctx.fillText(label, textX, textY);
        });

        return outputCanvas;
    }
}