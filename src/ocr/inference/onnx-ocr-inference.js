// onnx-ocr-inference.js

/**
 * @fileoverview Orquestador de inferencia OCR usando ONNX Runtime Web.
 */

// Ensure ONNX Runtime is loaded (usually via CDN)

// Importar los módulos JS que creamos antes
import { getOcrModel } from './hub.js';         // Ajusta la ruta si es necesario
import { AVAILABLE_OCR_MODELS } from './hub.js'; // Para validar nombres si es necesario
import { OCR_DEFAULT_CONFIG } from './config.js'; // O OCR_CONFIGS si usas el diccionario
import { OCRUtils } from './process.js';     // Donde definiste loadImageAndConvertToGrayscale, etc.

/**
 * Helper interno para procesar diversas fuentes de entrada de imagen.
 * Convierte las fuentes a un array de cv.Mat en escala de grises.
 * Requiere que OpenCV (cv) esté cargado globalmente y listo.
 *
 * @param {HTMLImageElement | HTMLCanvasElement | ImageData | cv.Mat | Array<HTMLImageElement | HTMLCanvasElement | ImageData | cv.Mat>} source - Fuente(s) de imagen.
 * @returns {Promise<cv.Mat[]>} Un array de objetos cv.Mat (escala de grises).
 * @throws {Error} Si OpenCV no está listo o no se pueden cargar imágenes válidas.
 */
async function _loadImageInputs(source) {
    // 'cv' debería estar disponible globalmente en el worker.
    if (typeof cv === 'undefined' || typeof cv.Mat === 'undefined') {
        throw new Error("OpenCV.js no está listo o no se encontró globalmente (cv).");
    }
    // const cv = window.cv; // INCORRECTO en worker

    const sources = Array.isArray(source) ? source : [source];
    const mats = [];
    const matsToDelete = [];

    console.debug(`OCR _loadImageInputs: Procesando ${sources.length} fuente(s)...`);

    for (let i = 0; i < sources.length; i++) {
        const src = sources[i];
        let grayMat = null;
        try {
            if (src instanceof cv.Mat) {
                if (src.empty()) {
                    console.warn(`OCR _loadImageInputs: Índice ${i}: cv.Mat vacío.`);
                    continue;
                }
                if (src.channels() === 1) {
                    grayMat = src.clone(); // Clonar para evitar modificar el original si es pasado
                    matsToDelete.push(grayMat);
                } else {
                    grayMat = new cv.Mat();
                    matsToDelete.push(grayMat);
                    if (src.channels() === 3) cv.cvtColor(src, grayMat, cv.COLOR_RGB2GRAY);
                    else if (src.channels() === 4) cv.cvtColor(src, grayMat, cv.COLOR_RGBA2GRAY);
                    else throw new Error(`Número de canales no soportado en cv.Mat: ${src.channels()}`);
                }
            } else if (src instanceof OffscreenCanvas || (typeof HTMLCanvasElement !== 'undefined' && src instanceof HTMLCanvasElement) || (typeof HTMLImageElement !== 'undefined' && src instanceof HTMLImageElement) || src instanceof ImageData) {
                 grayMat = OCRUtils.loadImageAndConvertToGrayscale(src);
                 if (grayMat) {
                      matsToDelete.push(grayMat);
                 }
            } else {
                console.warn(`OCR _loadImageInputs: Índice ${i}: Tipo no soportado: ${src ? src.constructor.name : src}.`);
            }
            if (grayMat && !grayMat.empty()) {
                mats.push(grayMat);
            }
        } catch (error) {
            console.error(`OCR _loadImageInputs: Error procesando fuente ${i}:`, error);
            if (grayMat && matsToDelete.includes(grayMat) && !grayMat.isDeleted()) {
                grayMat.delete();
                const indexToRemove = matsToDelete.indexOf(grayMat);
                if(indexToRemove > -1) matsToDelete.splice(indexToRemove, 1);
            }
        }
    }

    // Devolver solo los mats que no son el original, los otros se borran aquí.
    // O el llamador se encarga de borrar todos. Por ahora, devolvemos todos los procesados.
    // Es crucial que el llamador (OnnxOcrRecognizer.run) borre estos mats.
    return mats; // Estos son los mats grises, listos para preprocesar.
}


export class OnnxOcrRecognizer {
    session = null;
    config = null;
    modelName = null;
    providers = ['wasm'];
    sessionOptions = undefined;
    isInitialized = false;

    constructor(options = {}) {
        this.providers = options.providers || ['wasm']; // Aunque en el worker, se configurará via ort.env
        this.sessionOptions = options.sessionOptions;
        console.log(`OnnxOcrRecognizer (fast-plate-ocr-js): Creado. Proveedores solicitados: ${this.providers.join(', ')}`);
    }

    async initialize(modelName) {
        this.isInitialized = false;
        this.modelName = modelName;
        console.log(`OnnxOcrRecognizer: Inicializando con modelo: ${modelName}...`);

        try {
            if (typeof ort === 'undefined') { // Verificar que ort esté disponible globalmente
                throw new Error("ONNX Runtime (ort) no está disponible globalmente en el worker.");
            }
            if (!AVAILABLE_OCR_MODELS[modelName]) {
                throw new Error(`Modelo OCR '${modelName}' no en AVAILABLE_OCR_MODELS.`);
            }
            this.config = OCR_DEFAULT_CONFIG; // Asumiendo una config global
            if (!this.config) {
                throw new Error(`Configuración no encontrada para modelo: ${modelName}`);
            }
            const modelResponse = await getOcrModel(modelName);
            const modelArrayBuffer = await modelResponse.arrayBuffer();

            // Las sessionOptions se construyen aquí, usando this.sessionOptions si existen
            const finalSessionOptions = {
                executionProviders: this.providers, // Esto podría tomarse de ort.env.wasm.executionProviders si se prefiere una config global del worker
                logSeverityLevel: 2, // 0:verbose, 1:info, 2:warning, 3:error, 4:fatal
                ...(this.sessionOptions || {}) // Mezclar con opciones pasadas
            };
            console.log("OnnxOcrRecognizer: Creando sesión ONNX con opciones:", finalSessionOptions);
            this.session = await ort.InferenceSession.create(modelArrayBuffer, finalSessionOptions);

            console.log("OnnxOcrRecognizer: Sesión ONNX creada. Proveedores usados:", this.session.providers);
            this.isInitialized = true;
        } catch (error) {
            console.error(`OnnxOcrRecognizer: Fallo al inicializar para ${modelName}:`, error);
            this.session = null;
            this.config = null;
            this.modelName = null;
            this.isInitialized = false;
            throw error;
        }
    }

    async run(source, returnConfidence = false) {
        if (!this.isInitialized || !this.session || !this.config) {
            throw new Error("OnnxOcrRecognizer: No inicializado.");
        }
        // 'cv' debería estar disponible globalmente en el worker
        if (typeof cv === 'undefined' || !cv.imread) {
            throw new Error("OpenCV.js (cv) no está listo en OnnxOcrRecognizer.run.");
        }

        console.log("OnnxOcrRecognizer: Iniciando run...");
        let inputMats = [];
        try {
            inputMats = await _loadImageInputs(source); // Devuelve array de cv.Mat grises
            if (inputMats.length === 0) {
                console.warn("OnnxOcrRecognizer: Ninguna imagen válida de _loadImageInputs. Devolviendo vacío.");
                return returnConfidence ? [[], []] : [];
            }

            const preprocessed = OCRUtils.preprocessImage(inputMats, this.config.img_height, this.config.img_width);

            inputMats.forEach(mat => {
                if (mat && !mat.isDeleted()) mat.delete();
            }); // Borrar mats grises
            inputMats = [];

            const inputName = this.session.inputNames[0];
            if (!inputName) throw new Error("No se pudo determinar nombre de entrada del modelo OCR.");
            // Asegúrate que 'ort' aquí sea el global del worker.
            const inputTensor = new ort.Tensor('uint8', preprocessed.data, preprocessed.shape);
            const feeds = {[inputName]: inputTensor};
            const results = await this.session.run(feeds);
            const outputName = this.session.outputNames[0];
            if (!outputName) throw new Error("No se pudo determinar nombre de salida del modelo OCR.");
            const outputTensor = results[outputName];
            const finalResult = OCRUtils.postprocessOutput(
                outputTensor.data,
                outputTensor.dims,
                this.config.max_plate_slots,
                this.config.alphabet,
                returnConfidence
            );
            console.log("OnnxOcrRecognizer: Run completado.");
            return finalResult;
        } catch (error) {
            console.error("Error en OnnxOcrRecognizer.run:", error);
            if (inputMats && inputMats.length > 0) {
                inputMats.forEach(mat => {
                    if (mat && !mat.isDeleted()) mat.delete();
                });
            }
            throw error;
        }
    }
}