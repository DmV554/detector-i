// src/ocr/inference/onnx-ocr-inference.js

import { OCRUtils } from './process.js'; // Tus funciones de pre/postprocesamiento

export class OnnxOcrRecognizer {
    session = null;
    modelConfig = null; // Para guardar la configuración específica del modelo OCR (max_plate_slots, alphabet, etc.)
    modelPath = null;   // Ruta al archivo .onnx
    configPath = null;  // Ruta al archivo JSON de configuración del modelo OCR
    executionProviders = ['wasm']; // Default, se sobrescribirá desde opciones
    isInitialized = false;

    /**
     * @param {object} options
     * @param {string} options.modelPath - Ruta completa al archivo .onnx del modelo OCR.
     * @param {string} options.configPath - Ruta completa al archivo JSON de configuración del modelo OCR.
     * @param {string[]} [options.executionProviders=['wasm']] - Proveedores de ejecución para ONNX.
     * @param {object} [options.sessionOptions] - Opciones adicionales para la sesión ONNX.
     */
    constructor(options = {}) {
        if (!options.modelPath || !options.configPath) {
            throw new Error("OnnxOcrRecognizer: 'modelPath' y 'configPath' son requeridos en las opciones.");
        }
        this.modelPath = options.modelPath;
        this.configPath = options.configPath;
        this.executionProviders = options.executionProviders || ['wasm'];
        this.sessionOptions = options.sessionOptions; // Opciones adicionales de sesión (ej. logSeverityLevel)
        console.log(`OnnxOcrRecognizer: Creado. Modelo: ${this.modelPath}, Config: ${this.configPath}, EPs: ${this.executionProviders.join(', ')}`);
    }

    async initialize() {
        if (this.isInitialized) {
            console.log("OnnxOcrRecognizer ya está inicializado.");
            return;
        }
        console.log(`OnnxOcrRecognizer: Inicializando con modelo: ${this.modelPath} y config: ${this.configPath}...`);

        try {
            if (typeof ort === 'undefined') {
                throw new Error("ONNX Runtime (ort) no está disponible globalmente.");
            }

            // 1. Cargar el archivo JSON de configuración específico del modelo OCR
            const configResponse = await fetch(this.configPath);
            if (!configResponse.ok) {
                throw new Error(`Error HTTP ${configResponse.status} cargando config OCR desde ${this.configPath}`);
            }
            this.modelConfig = await configResponse.json();
            if (!this.modelConfig || !this.modelConfig.alphabet || !this.modelConfig.max_plate_slots) {
                throw new Error(`Configuración OCR inválida o incompleta en ${this.configPath}. Faltan 'alphabet' o 'max_plate_slots'.`);
            }
            console.log("OnnxOcrRecognizer: Configuración del modelo OCR cargada:", this.modelConfig);


            // 2. Cargar el modelo ONNX
            //    getOcrModel de hub.js ya no se usa. Se carga directamente el ArrayBuffer.
            const modelResponse = await fetch(this.modelPath);
            if (!modelResponse.ok) {
                throw new Error(`Error HTTP ${modelResponse.status} cargando modelo ONNX desde ${this.modelPath}`);
            }
            const modelArrayBuffer = await modelResponse.arrayBuffer();

            const finalSessionOptions = {
                executionProviders: this.executionProviders,
                logSeverityLevel: 2, // 0:verbose, 1:info, 2:warning, 3:error, 4:fatal
                ...(this.sessionOptions || {}) // Mezclar con opciones pasadas si existen
            };
            console.log("OnnxOcrRecognizer: Creando sesión ONNX con opciones:", finalSessionOptions);

            this.session = await ort.InferenceSession.create(modelArrayBuffer, finalSessionOptions);
            console.log("OnnxOcrRecognizer: Sesión ONNX creada. Proveedores efectivos usados:", this.session.providers);
            this.isInitialized = true;

        } catch (error) {
            console.error(`OnnxOcrRecognizer: Fallo al inicializar:`, error, error.stack);
            this.session = null;
            this.modelConfig = null;
            this.isInitialized = false;
            throw error; // Re-lanzar para que el llamador lo maneje
        }
    }

    /**
     * Ejecuta la inferencia OCR.
     * @param {cv.Mat | HTMLImageElement | HTMLCanvasElement | ImageData | OffscreenCanvas} source - Imagen de entrada.
     * @param {boolean} [returnConfidence=false] - Si se deben devolver las confianzas.
     * @returns {Promise<[string[], (number[][]|undefined)] | string[]>}
     * Si returnConfidence es true, devuelve [arrayDeTextos, arrayDeConfianzasPorSlot].
     * Sino, devuelve arrayDeTextos.
     */
    async run(source, returnConfidence = false) {
        if (!this.isInitialized || !this.session || !this.modelConfig) {
            throw new Error("OnnxOcrRecognizer: No inicializado o configuración de modelo no cargada.");
        }
        if (typeof cv === 'undefined' || !cv.imread) {
            throw new Error("OpenCV.js (cv) no está listo en OnnxOcrRecognizer.run.");
        }

        console.debug("OnnxOcrRecognizer: Iniciando run...");
        let grayMat = null; // El mat en escala de grises
        let preprocessedInput = null;

        try {
            // 1. Cargar y convertir a escala de grises
            // _loadImageInputs procesaba un array, aquí procesamos una sola imagen.
            // loadImageAndConvertToGrayscale espera una sola fuente y devuelve un solo Mat gris.
            grayMat = OCRUtils.loadImageAndConvertToGrayscale(source);
            if (!grayMat || grayMat.empty()) {
                console.warn("OnnxOcrRecognizer: No se pudo obtener un Mat gris válido de la fuente.");
                return returnConfidence ? [[], undefined] : [];
            }

            // 2. Preprocesar para el modelo (redimensionar, normalizar, formatear tensor)
            //    Pasamos [grayMat] porque preprocessOcrInputs espera un array.
            preprocessedInput = OCRUtils.preprocessOcrInputs(
                [grayMat], // Espera un array de Mats
                this.modelConfig.img_height,
                this.modelConfig.img_width,
                this.modelConfig.expected_input_channels || 1 // Tomar de config o default a 1
            );

            // 3. Crear tensor de entrada
            const inputName = this.session.inputNames[0];
            if (!inputName) throw new Error("No se pudo determinar nombre de entrada del modelo OCR.");
            const inputTensor = new ort.Tensor('float32', preprocessedInput.data, preprocessedInput.shape);
            const feeds = { [inputName]: inputTensor };

            // 4. Ejecutar inferencia
            const results = await this.session.run(feeds);
            const outputName = this.session.outputNames[0];
            if (!outputName) throw new Error("No se pudo determinar nombre de salida del modelo OCR.");
            const outputTensor = results[outputName];

            // 5. Postprocesar salida
            const finalResult = OCRUtils.postprocessOcrOutput(
                outputTensor.data,      // Float32Array o similar
                outputTensor.dims,      // shape
                this.modelConfig.max_plate_slots,
                this.modelConfig.alphabet,
                returnConfidence
            );

            console.debug("OnnxOcrRecognizer: Run completado.");
            // La salida de postprocessOcrOutput ya es [plates] o [plates, probabilities]
            // Tu DefaultOCR espera un objeto { textArray: [...], probabilities: { mean: ...}}
            // Así que necesitamos adaptar esto.
            if (returnConfidence) {
                const plates = finalResult[0];
                const probArrays = finalResult[1]; // Array de arrays de probabilidades
                // Calcular una confianza media por placa si es necesario, o pasar el array
                const confidenceMeans = probArrays ? probArrays.map(pArr => pArr.reduce((a,b)=>a+b,0) / (pArr.length || 1)) : [];
                 // Devolver en el formato que espera DefaultOCR
                return {
                    textArray: plates, // plates es un array de strings, ej. ["ABC123"]
                    probabilities: { mean: () => (confidenceMeans[0] || 0) } // Tomar la media de la primera (y única) placa
                };

            } else {
                const plates = finalResult; // plates es un array de strings
                 return {
                    textArray: plates,
                    probabilities: { mean: () => 0 } // O alguna estructura de confianza placeholder
                };
            }

        } catch (error) {
            console.error("Error en OnnxOcrRecognizer.run:", error);
            throw error; // Re-lanzar para que DefaultOCR lo maneje
        } finally {
            if (grayMat && !grayMat.isDeleted()) grayMat.delete();
            // preprocessedInput.data es un Float32Array, no necesita delete.
        }
    }
}