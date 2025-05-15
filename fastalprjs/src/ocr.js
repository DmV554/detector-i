/**
 * @fileoverview Implementación por defecto de OCR, usando OnnxOcrRecognizer.
 */

// Importar clases/interfaces base de nuestra nueva librería
import { BaseOcr, OcrResult } from './base.js';

// Importar la implementación CONCRETA del OCR que vamos a envolver
// ¡¡ASEGÚRATE QUE ESTA RUTA SEA CORRECTA!!
import { OnnxOcrRecognizer } from '/detector-i/fast-plate-ocr-js/inference/onnx-ocr-inference.js'; // Ajusta la ruta!

// Opcional: importar tipos para documentación
// import { OcrModelName } from './hub-ocr.js';
// import * as ort from 'onnxruntime-web';

/**
 * Implementación por defecto de BaseOcr que utiliza la clase
 * OnnxOcrRecognizer existente internamente.
 *
 * @export
 * @class DefaultOcr
 * @extends {BaseOcr}
 */
export class DefaultOcr extends BaseOcr {
    /**
     * Instancia interna del reconocedor envuelto.
     * @type {OnnxOcrRecognizer | null}
     * @private
     */
    _recognizerInstance = null;

    /** @type {string} Nombre del modelo OCR a cargar. */
    modelName;
    /** @type {string[] | undefined} Proveedores para ONNX Runtime. */
    providers;
    /** @type {ort.SessionOptions | undefined} Opciones de sesión para ONNX Runtime. */
    sessionOptions;

    /**
     * Configura el DefaultOcr. La inicialización real (carga del modelo)
     * ocurre de forma asíncrona al llamar a `initialize()`.
     *
     * @param {object} [options={}] - Opciones de configuración.
     * @param {string} [options.modelName='ocr-latin-v1'] - Nombre del modelo OCR (debe existir en hub-ocr.js). Reemplaza el default si es necesario.
     * @param {string[]} [options.providers] - Proveedores de ejecución ONNX Runtime opcionales (ej. ['wasm']). Si no se especifica, usa los defaults de OnnxOcrRecognizer.
     * @param {any} [options.sessionOptions] - Opciones de sesión ONNX Runtime opcionales. Si no se especifica, usa los defaults de OnnxOcrRecognizer.
     */
    constructor(options = {}) {
        super(); // Llama al constructor de BaseOcr
        // TODO: Considera obtener el nombre default desde hub-ocr.js si es posible
        this.modelName = options.modelName || 'ocr-latin-v1'; // Usa un default o el proporcionado
        this.providers = options.providers;
        this.sessionOptions = options.sessionOptions;
        this._recognizerInstance = null; // Marcar como no inicializado

        console.log(`DefaultOcr configurado: Modelo=${this.modelName}`);
    }

    /**
     * Inicializa el reconocedor OCR subyacente (`OnnxOcrRecognizer`) de forma asíncrona.
     * Debe llamarse antes de usar `predict`.
     * @override // Indica que estamos sobrescribiendo un método de BaseOcr
     * @returns {Promise<void>}
     * @throws {Error} Si la carga del modelo o la creación de la sesión falla.
     */
    async initialize() {
        if (this._recognizerInstance) {
            console.warn("El reconocedor OCR subyacente ya está inicializado.");
            return;
        }
        console.log(`Inicializando el OnnxOcrRecognizer subyacente (${this.modelName})...`);
        try {
            // 1. Crear la instancia del reconocedor que envolvemos
            this._recognizerInstance = new OnnxOcrRecognizer({
                providers: this.providers,
                sessionOptions: this.sessionOptions
            });
            // 2. Inicializarla (esto carga el modelo y crea la sesión ONNX)
            await this._recognizerInstance.initialize(this.modelName);
            console.log("OnnxOcrRecognizer subyacente inicializado correctamente.");
        } catch (error) {
            console.error(`Fallo al inicializar OnnxOcrRecognizer (${this.modelName}):`, error);
            this._recognizerInstance = null; // Asegurar que esté null si falla
            throw error; // Relanzar el error
        }
    }

    /**
     * Realiza OCR en la imagen recortada usando la instancia envuelta
     * de `OnnxOcrRecognizer` y transforma el resultado al formato `OcrResult`.
     * NOTA: La conversión a escala de grises es manejada internamente por `OnnxOcrRecognizer.run`.
     *
     * @override // Indica que estamos sobrescribiendo un método de BaseOcr
     * @param {ImageData | HTMLImageElement | HTMLCanvasElement | Blob | File | cv.Mat} croppedPlateSource - La imagen recortada de la matrícula.
     * @returns {Promise<OcrResult | null>} Una promesa que se resuelve con un objeto OcrResult (definido en base.js) o null si la predicción falla o no devuelve resultados válidos.
     * @throws {Error} Si el reconocedor no está inicializado.
     */
    async predict(croppedPlateSource) {
        if (!this._recognizerInstance || !this._recognizerInstance.isInitialized) {
            throw new Error("El DefaultOcr (OnnxOcrRecognizer) no ha sido inicializado. Llama a initialize() primero.");
        }
        // Validar entrada básica
        if (!croppedPlateSource) {
             console.warn("DefaultOcr.predict llamada con croppedPlateSource nulo o undefined. Devolviendo null.");
             return null;
        }

        console.debug("Llamando a run() del OnnxOcrRecognizer envuelto...");
        try {
            // Llama al método run de la instancia envuelta, pidiendo confianzas
            // El resultado esperado es [array_de_textos, array_de_arrays_de_confianzas]
            const results = await this._recognizerInstance.run(croppedPlateSource, true);

            // Validar la estructura del resultado devuelto por el recognizer envuelto
            if (!results || !Array.isArray(results) || results.length !== 2 ||
                !Array.isArray(results[0]) || !Array.isArray(results[1]) ||
                results[0].length === 0 || results[1].length === 0) {
                console.warn("La predicción OCR envuelta no devolvió la estructura esperada ([texts], [confidences]) o estaba vacía. Results:", results);
                return null; // No se pudo obtener un resultado válido
            }

            // Asumimos que procesamos una sola imagen, tomamos el primer resultado
            const plateText = results[0][0]; // El primer string de texto reconocido
            const charConfidences = results[1][0]; // El primer array de confianzas por caracter

            if (typeof plateText !== 'string' || !Array.isArray(charConfidences)) {
                 console.warn("El texto o las confianzas recibidas del recognizer envuelto no tienen el tipo esperado.", plateText, charConfidences);
                 return null;
            }

            // --- Transformación a OcrResult ---

            // 1. Limpiar el texto (eliminar caracter de padding si existe)
            const padChar = this._recognizerInstance.config?.pad_char; // Obtener pad_char de la config cargada
            let cleanedText = plateText;
            if (padChar && typeof padChar === 'string' && padChar.length > 0) {
                try {
                    // Escapar caracteres especiales de regex en padChar antes de crear RegExp
                    const escapedPadChar = padChar.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
                    const padCharRegex = new RegExp(escapedPadChar, 'g');
                    cleanedText = plateText.replace(padCharRegex, '');
                    console.debug(`Texto original: "${plateText}", Padding ('${padChar}') quitado: "${cleanedText}"`);
                } catch (regexError) {
                     console.error("Error al crear regex para quitar padding, usando texto original:", regexError);
                     // Continuar con el texto original si falla el regex
                }
            }

            // 2. Calcular confianza promedio
            let meanConfidence = 0;
            // Filtrar solo números válidos en el array de confianzas
            const validConfidences = charConfidences.filter(c => typeof c === 'number' && isFinite(c));
            if (validConfidences.length > 0) {
                 meanConfidence = validConfidences.reduce((sum, conf) => sum + conf, 0) / validConfidences.length;
            } else {
                 console.warn("No se encontraron confianzas válidas por caracter para calcular el promedio.");
                 // ¿Devolver 0, NaN, null? Devolvemos 0 como default.
                 meanConfidence = 0;
            }

            // 3. Crear y devolver la instancia de OcrResult (de base.js)
            return new OcrResult(cleanedText, meanConfidence); // Usamos la confianza promedio

        } catch (error) {
            console.error("Error durante la predicción OCR envuelta:", error);
            // Decidir si relanzar el error o devolver null
            // throw error; // Opción 1: Relanzar
            return null; // Opción 2: Devolver null indica fallo en la predicción
        }
    }
}