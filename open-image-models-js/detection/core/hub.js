/**
 * @fileoverview Model Hub for loading local models in the browser.
 * Loads models from a relative path ('./models/') instead of fetching from external URLs.
 */

/**
 * @typedef {'yolo-v9-s-608-license-plate-end2end' |
 * 'yolo-v9-t-640-license-plate-end2end' |
 * 'yolo-v9-t-512-license-plate-end2end' |
 * 'yolo-v9-t-416-license-plate-end2end' |
 * 'yolo-v9-t-384-license-plate-end2end' |
 * 'yolo-v9-t-256-license-plate-end2end'} PlateDetectorModelName
 * Available ONNX model names for license plate detection.
 */

/**
 * Dictionary mapping available ONNX model names to their LOCAL relative paths.
 * **IMPORTANT:** Ensure these files exist in a 'models' subdirectory relative
 * to where your HTML file is served.
 * @type {Record<PlateDetectorModelName, string>}
 */
export const AVAILABLE_ONNX_MODELS = {
    // Plate Detection - Paths are now relative
    "yolo-v9-s-608-license-plate-end2end": '/models/yolo-v9-s-608-license-plates-end2end.onnx',
    "yolo-v9-t-640-license-plate-end2end": '/models/yolo-v9-t-640-license-plates-end2end.onnx',
    "yolo-v9-t-512-license-plate-end2end": '/models/yolo-v9-t-512-license-plates-end2end.onnx',
    "yolo-v9-t-416-license-plate-end2end": '/models/yolo-v9-t-416-license-plates-end2end.onnx',
    "yolo-v9-t-384-license-plate-end2end": '/models/yolo-v9-t-384-license-plates-end2end.onnx', // Asegúrate que este archivo exista localmente
    "yolo-v9-t-256-license-plate-end2end": '/models/yolo-v9-t-256-license-plates-end2end.onnx',
};


/**
 * Fetches a detection model from a LOCAL relative path ('./models/').
 * Returns the Response object, allowing the caller to get ArrayBuffer, Blob, etc.
 * Does NOT use Cache Storage API anymore; relies on standard browser HTTP caching if applicable.
 *
 * @export
 * @param {PlateDetectorModelName} modelName - Which model to load.
 * @param {object} [options={}] - Options object (currently unused, kept for signature compatibility perhaps).
 * @returns {Promise<Response>} A promise resolving to the Response object containing the model data from the local path.
 * @throws {Error} If the model name is invalid or fetching from the local path fails.
 */
export async function getModel(modelName, options = {}) {
    // Options like forceDownload, cacheName, onProgress are no longer relevant here.
    // const { forceDownload, cacheName, onProgress } = options; // Removed

    if (!(modelName in AVAILABLE_ONNX_MODELS)) {
        const availableModels = Object.keys(AVAILABLE_ONNX_MODELS).join(", ");
        // Consider throwing a more specific Error type if desired
        throw new Error(`Unknown model name: ${modelName}. Use one of [${availableModels}]`);
    }

    const modelLocalPath = AVAILABLE_ONNX_MODELS[modelName];
    console.log(`Attempting to fetch model '${modelName}' from local path: ${modelLocalPath}...`);

    // Use fetch to get the local file.
    // Since the path is relative and served from the same origin, CORS is not an issue.
    const request = new Request(modelLocalPath);
    let response;
    try {
        response = await fetch(request);
    } catch (fetchError) {
         console.error(`Network error while fetching local model ${modelLocalPath}:`, fetchError);
         throw new Error(`Network error fetching local model '${modelName}': ${fetchError.message}`);
    }

    if (!response.ok) {
        console.error(`Failed to fetch local model ${modelLocalPath}. Status: ${response.status} ${response.statusText}`);
        throw new Error(`Failed to fetch local model '${modelName}' from path ${modelLocalPath}. Server responded with status ${response.status}. Ensure the file exists and the server is configured correctly.`);
    }

    console.log(`Successfully fetched model '${modelName}' from local path.`);
    // Return the Response object (body not consumed yet)
    return response;
}