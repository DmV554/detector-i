/**
 * @fileoverview Model Hub for loading local OCR models in the browser.
 * Loads models from a relative path ('./models-ocr/').
 */

/**
 * @typedef {'ocr-latin-v1'} OcrModelName

/**
 * Dictionary mapping available OCR ONNX model names to their LOCAL relative paths.
 * **IMPORTANT:** Ensure these files exist in a 'models-ocr' subdirectory relative
 * to where your HTML file is served.
 * @type {Record<OcrModelName, string>}
 */
export const AVAILABLE_OCR_MODELS = {
    "global-plates-mobile-vit-v2-model": '/models-ocr/global_mobile_vit_v2_ocr.onnx',
};


/**
 * Fetches an OCR model from a LOCAL relative path ('./models-ocr/').
 * Returns the Response object, allowing the caller to get ArrayBuffer, Blob, etc.
 * Relies on standard browser HTTP caching if applicable.
 *
 * @export
 * @param {OcrModelName} modelName - Which OCR model to load.
 * @param {object} [options={}] - Options object (actualmente no se usa, se mantiene por posible compatibilidad de firma).
 * @returns {Promise<Response>} Una promesa que se resuelve con el objeto Response que contiene los datos del modelo.
 * @throws {Error} Si el nombre del modelo es inválido o falla la obtención desde la ruta local.
 */
export async function getOcrModel(modelName, options = {}) {
    if (!(modelName in AVAILABLE_OCR_MODELS)) {
        const availableModels = Object.keys(AVAILABLE_OCR_MODELS).join(", ");
        // Considera lanzar un Error más específico si lo deseas
        throw new Error(`Nombre de modelo OCR desconocido: ${modelName}. Usa uno de [${availableModels}]`);
    }

    const modelLocalPath = AVAILABLE_OCR_MODELS[modelName];
    console.log(`Intentando obtener modelo OCR '${modelName}' desde ruta local: ${modelLocalPath}...`);

    const request = new Request(modelLocalPath);
    let response;
    try {
        response = await fetch(request);
    } catch (fetchError) {
         console.error(`Error de red al obtener el modelo OCR local ${modelLocalPath}:`, fetchError);
         throw new Error(`Error de red al obtener el modelo OCR local '${modelName}': ${fetchError.message}`);
    }

    if (!response.ok) {
        console.error(`Fallo al obtener el modelo OCR local ${modelLocalPath}. Estado: ${response.status} ${response.statusText}`);
        throw new Error(`Fallo al obtener el modelo OCR local '${modelName}' desde la ruta ${modelLocalPath}. El servidor respondió con estado ${response.status}. Asegúrate que el archivo existe y el servidor está configurado correctamente.`);
    }

    console.log(`Modelo OCR '${modelName}' obtenido exitosamente desde ruta local.`);
    return response;
}
