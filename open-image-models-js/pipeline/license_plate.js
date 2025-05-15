/**
 * @fileoverview License Plate detection pipeline using YoloV9.
 * Equivalent structure to open_image_models/detection/pipeline/license_plate.py
 */

// --- Imports ---
// Adjust paths based on your actual file structure

// Import the base detector class we are extending
import { YoloV9ObjectDetector } from '../detection/core/yolo_v9/inference.js';
// Import the function to fetch model data (browser cache aware) and the model name type
import { getModel, AVAILABLE_ONNX_MODELS } from '../detection/core/hub.js';
// Import base types if needed for documentation or type checking (optional here)
import { DetectionResult } from '../detection/core/base.js';

// Type definition for model names (can also be imported from hub.js if exported there)
/**
 * @typedef {'yolo-v9-s-608-license-plate-end2end' |
 * 'yolo-v9-t-640-license-plate-end2end' |
 * 'yolo-v9-t-512-license-plate-end2end' |
 * 'yolo-v9-t-416-license-plate-end2end' |
 * 'yolo-v9-t-384-license-plate-end2end' |
 * 'yolo-v9-t-256-license-plate-end2end'} PlateDetectorModelName
 */


// Basic logger using console
const LOGGER = {
    info: (...args) => console.log('[INFO]', ...args),
    warn: (...args) => console.warn('[WARN]', ...args),
    error: (...args) => console.error('[ERROR]', ...args),
};


/**
 * Specialized detector for license plates using a YoloV9 model.
 * Simplifies setup by fetching the appropriate model and setting LP-specific labels.
 * Use the static `LicensePlateDetector.create()` method to instantiate.
 *
 * @export
 * @class LicensePlateDetector
 * @extends {YoloV9ObjectDetector}
 */
export class LicensePlateDetector extends YoloV9ObjectDetector {

    /**
     * Private constructor. Use the static async `create` method instead.
     * Initializes the superclass with fetched model data and license plate specific config.
     * @private
     * @param {PlateDetectorModelName} detectionModelName - Name used for logging/identification.
     * @param {ArrayBuffer|Uint8Array|Blob} modelData - The ONNX model data.
     * @param {object} [options={}] - Configuration options passed from `create`.
     * @param {number} [options.confThresh=0.25] - Confidence threshold.
     * @param {string[]} [options.providers=['wasm']] - Execution providers.
     * @param {ort.SessionOptions} [options.sessionOptions] - Advanced session options.
     */
    constructor(detectionModelName, modelData, options = {}) {
        const {
            confThresh, // Will default in super if undefined
            providers,  // Will default in super if undefined
            sessionOptions // Will default in super if undefined
        } = options;

        // Call the parent constructor with the fetched model data,
        // hardcoded license plate labels, and configuration options.
        super(modelData, ["License Plate"], { // Hardcoded class label for LP detection
            confThresh: confThresh,
            providers: providers,
            sessionOptions: sessionOptions,
            modelName: detectionModelName || 'license-plate-detector' // Pass model name for logging
        });

        // Logging moved to the 'create' method after successful initialization
    }

    /**
     * Asynchronously creates and initializes a LicensePlateDetector instance.
     * Fetches the specified model using the hub's getModel function.
     *
     * @static
     * @async
     * @param {PlateDetectorModelName} detectionModel - Name of the license plate detection model to use.
     * @param {object} [options={}] - Configuration options.
     * @param {number} [options.confThresh=0.25] - Confidence threshold for filtering predictions.
     * @param {string[]} [options.providers=['wasm']] - Execution providers for ONNX Runtime (e.g., ['wasm', 'webgl']).
     * @param {ort.SessionOptions} [options.sessionOptions] - Advanced ONNX Runtime session options. Provider options should be set here.
     * @param {boolean} [options.forceDownload=false] - Force download model even if cached.
     * @returns {Promise<LicensePlateDetector>} A promise that resolves with the initialized detector instance.
     * @throws {Error} If model fetching or initialization fails.
     */
    static async create(detectionModel, options = {}) {
        LOGGER.info(`Creating LicensePlateDetector with model: ${detectionModel}`);

        // 1. Fetch the model data (using browser cache via getModel)
        let modelData;
        try {
            const modelResponse = await getModel(detectionModel, {
                 forceDownload: options.forceDownload ?? false
                 // Pass other getModel options if needed (e.g., cacheName, onProgress)
            });
            modelData = await modelResponse.arrayBuffer();
             LOGGER.info(`Model data for ${detectionModel} fetched successfully (${(modelData.byteLength / (1024*1024)).toFixed(2)} MB).`);
        } catch (error) {
            LOGGER.error(`Failed to fetch model data for ${detectionModel}:`, error);
            throw error; // Re-throw error after logging
        }

        // 2. Create the instance using the private constructor
        // Pass only relevant options for the superclass constructor
        const instance = new LicensePlateDetector(detectionModel, modelData, {
            confThresh: options.confThresh,
            providers: options.providers,
            sessionOptions: options.sessionOptions,
            // Pass detectionModel name again for logging inside constructor/super
            detectionModelName: detectionModel
        });

        // 3. Ensure the underlying ONNX session is initialized (await the promise set up by super constructor)
        try {
            await instance._ensureInitialized(); // Wait for the async init in the superclass
            LOGGER.info(`LicensePlateDetector with model ${detectionModel} initialized successfully.`);
        } catch (initError) {
             LOGGER.error(`Failed to complete initialization for ${detectionModel}:`, initError);
             throw initError; // Re-throw initialization error
        }


        // 4. Return the fully initialized instance
        return instance;
    }


    /**
     * Perform license plate detection on one or multiple image sources.
     * This method simply calls the `predict` method of the parent `YoloV9ObjectDetector` class.
     *
     * @override // Overrides method from ObjectDetectorBase (via YoloV9ObjectDetector)
     * @param {HTMLImageElement|HTMLCanvasElement|ImageBitmap|ImageData|HTMLVideoElement | (HTMLImageElement|HTMLCanvasElement|ImageBitmap|ImageData|HTMLVideoElement)[]} imageSources - A single image source or a list of image sources compatible with the Canvas API.
     * @returns {Promise<DetectionResult[] | DetectionResult[][]>} A list of detections for a single input, or a list of lists for multiple inputs. Each detection focuses on license plates.
     *
     * @example
     * async function detectPlates() {
     * try {
     * const lpDetector = await LicensePlateDetector.create('yolo-v9-t-384-license-plate-end2end');
     * const imgElement = document.getElementById('myCarImage'); // Ensure image is loaded
     * const detections = await lpDetector.predict(imgElement);
     * console.log("Detected License Plates:", detections);
     * } catch (error) {
     * console.error("License plate detection failed:", error);
     * }
     * }
     */
    async predict(imageSources) {
        // No changes needed here, the parent class handles the prediction logic
        // after being initialized with the correct model and class labels.
        return super.predict(imageSources);
    }

    // Inherits showBenchmark and displayPredictions from YoloV9ObjectDetector
    // No need to redefine unless specific modifications for license plates are needed.

}