// onnx-ocr-inference.js

/**
 * @fileoverview Orquestador de inferencia OCR usando ONNX Runtime Web.
 */

// Ensure ONNX Runtime is loaded (usually via CDN)
const ort = window.ort;
if (!ort) {
    throw new Error("ONNX Runtime Web (ort) not found. Ensure it's loaded, e.g., via CDN.");
}

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
    if (!window.cv || !window.cv.Mat) { // Verifica que cv esté en el scope global y tenga Mat
        throw new Error("OpenCV.js no está listo o no se encontró en window.cv.");
    }
    const cv = window.cv; // Accede a cv desde el scope global

    // Asegurar que la entrada sea un array para procesarla uniformemente
    const sources = Array.isArray(source) ? source : [source];
    const mats = [];
    const matsToDelete = []; // Para llevar registro de los Mats creados aquí

    console.debug(`Procesando ${sources.length} fuente(s) de imagen...`);

    for (let i = 0; i < sources.length; i++) {
        const src = sources[i];
        let grayMat = null;

        try {
            if (src instanceof cv.Mat) {
                if (src.empty()) {
                    console.warn(`Índice ${i}: Se proporcionó un cv.Mat vacío.`);
                    continue;
                }
                // Si ya es un Mat, verificar si es escala de grises
                if (src.channels() === 1) {
                    // Usar directamente (¿clonar para evitar efectos secundarios?)
                    // Por ahora, lo usamos directamente. El preprocesamiento lo redimensionará.
                    grayMat = src;
                    console.debug(`Índice ${i}: Usando cv.Mat proporcionado directamente (asumido escala de grises).`);
                } else {
                    // Intentar convertir si no es escala de grises
                    console.warn(`Índice ${i}: cv.Mat proporcionado tiene ${src.channels()} canales. Intentando convertir a escala de grises.`);
                    grayMat = new cv.Mat();
                    matsToDelete.push(grayMat); // Marcar para borrar más tarde
                    if (src.channels() === 3) cv.cvtColor(src, grayMat, cv.COLOR_RGB2GRAY);
                    else if (src.channels() === 4) cv.cvtColor(src, grayMat, cv.COLOR_RGBA2GRAY);
                    else throw new Error(`Número de canales no soportado en cv.Mat: ${src.channels()}`);
                }
            } else if (src instanceof HTMLImageElement || src instanceof HTMLCanvasElement || src instanceof ImageData) {
                // Convertir desde fuente de imagen del navegador
                 console.debug(`Índice ${i}: Procesando ${src.constructor.name}...`);
                 grayMat = OCRUtils.loadImageAndConvertToGrayscale(src); // Esta función ya devuelve Mat gris
                 if (grayMat) {
                      matsToDelete.push(grayMat); // Marcar para borrar más tarde
                 } else {
                      console.warn(`Índice ${i}: loadImageAndConvertToGrayscale devolvió null.`);
                 }
            } else {
                console.warn(`Índice ${i}: Tipo de entrada no soportado: ${typeof src}. Se omitirá.`);
                // Podrías añadir manejo para Blob/File aquí si lo necesitas
                // (creando un Image element temporalmente).
            }

            if (grayMat && !grayMat.empty()) {
                mats.push(grayMat);
            } else {
                 console.warn(`Índice ${i}: No se pudo obtener un cv.Mat válido en escala de grises.`);
            }

        } catch (error) {
            console.error(`Error procesando la fuente de imagen en el índice ${i}:`, error);
            // Limpiar el mat si se creó en este intento fallido
            if (grayMat && matsToDelete.includes(grayMat) && grayMat.delete) {
                grayMat.delete();
                const indexToRemove = matsToDelete.indexOf(grayMat);
                if(indexToRemove > -1) matsToDelete.splice(indexToRemove, 1);
            }
            // ¿Continuar con los demás o lanzar un error general? Por ahora continuamos.
        }
    } // Fin del bucle for

    if (mats.length === 0 && sources.length > 0) {
        throw new Error("No se pudieron cargar imágenes válidas desde la(s) fuente(s) proporcionada(s).");
    }

    // Devolver los mats válidos. Quien llama es responsable de borrarlos después de usarlos.
    // OJO: Si se pasó un Mat original, no está en matsToDelete y no se borrará aquí.
    console.debug(`Se obtuvieron ${mats.length} cv.Mat válidos.`);
    return mats;
}


export class OnnxOcrRecognizer {
    /** @type {ort.InferenceSession | null} */
    session = null;
    /** @type {import('./config-ocr.js').PlateOcrConfig | null} */
    config = null;
    /** @type {string | null} */
    modelName = null;
    /** @type {string[]} */
    providers = ['wasm']; // Proveedor por defecto
    /** @type {ort.SessionOptions | undefined} */
    sessionOptions = undefined; // Opciones de sesión ORT
     /** @type {boolean} */
    isInitialized = false;

    /**
     * @param {object} [options={}] Opciones de configuración.
     * @param {string[]} [options.providers=['wasm']] - Proveedores de ejecución para ONNX Runtime Web ('wasm', 'webgl', 'webgpu'). El orden importa.
     * @param {ort.SessionOptions} [options.sessionOptions] - Opciones avanzadas de sesión de ONNX Runtime.
     */
    constructor(options = {}) {
        this.providers = options.providers || ['wasm'];
        this.sessionOptions = options.sessionOptions; // Puede ser undefined
        console.log(`OnnxOcrRecognizer creado. Proveedores solicitados: ${this.providers.join(', ')}`);
    }

    /**
     * Carga el modelo OCR especificado y prepara la sesión de inferencia.
     * Debe llamarse antes de `run` o `benchmark`. Es asíncrona.
     * @param {import('./hub-ocr.js').OcrModelName} modelName - Nombre del modelo OCR a cargar (debe estar en hub-ocr.js).
     * @throws {Error} Si falla la carga del modelo, configuración o creación de la sesión.
     */
    async initialize(modelName) {
         this.isInitialized = false;
         this.modelName = modelName;
         console.log(`Inicializando OCR Recognizer con el modelo: ${modelName}...`);

        try {
            // 1. Validar Nombre del Modelo (Opcional pero bueno)
            if (!AVAILABLE_OCR_MODELS[modelName]) {
                 throw new Error(`El nombre del modelo OCR '${modelName}' no se encuentra en AVAILABLE_OCR_MODELS.`);
            }

            // 2. Obtener Configuración del Modelo
            // TODO: Adaptar si usas el diccionario OCR_CONFIGS en lugar del default
            this.config = OCR_DEFAULT_CONFIG;
            if (!this.config) {
                // Si usas el diccionario, sería: this.config = OCR_CONFIGS[modelName];
                throw new Error(`Configuración no encontrada para el modelo: ${modelName}`);
            }
            console.log("Configuración del modelo cargada:", this.config);

            // 3. Obtener Datos del Modelo (ArrayBuffer)
            console.log(`Obteniendo datos del modelo para ${modelName}...`);
            const modelResponse = await getOcrModel(modelName); // Desde hub-ocr.js
            const modelArrayBuffer = await modelResponse.arrayBuffer();
            console.log(`Datos del modelo obtenidos (${(modelArrayBuffer.byteLength / (1024*1024)).toFixed(2)} MB).`);

            // 4. Crear Sesión de Inferencia ONNX Runtime
            console.log(`Creando sesión ONNX Runtime con proveedores: ${this.providers.join(', ')}...`);
            // Usamos un objeto de opciones que incluye tanto providers como sessionOptions
            const sessionCreateOptions = {
                 executionProviders: this.providers,
                 sessionOptions: this.sessionOptions // Pasamos las opciones de sesión aquí
            };
            this.session = await ort.InferenceSession.create(modelArrayBuffer, sessionCreateOptions);
            console.log("Sesión ONNX Runtime creada exitosamente.");
            console.log("Proveedores realmente usados:", this.session.providers); // Muestra los que ORT pudo usar
            console.log("Nombres de entrada:", this.session.inputNames);
            console.log("Nombres de salida:", this.session.outputNames);

            // Verificar si los nombres de entrada/salida son los esperados (opcional)
             if (!this.session.inputNames.length || !this.session.outputNames.length) {
                  console.warn("El modelo cargado no parece tener nombres de entrada/salida definidos.");
             }

            this.isInitialized = true;
            console.log(`Reconocedor inicializado exitosamente para el modelo ${modelName}.`);

        } catch (error) {
            console.error(`Fallo al inicializar OnnxOcrRecognizer para el modelo ${modelName}:`, error);
            this.session = null;
            this.config = null;
            this.modelName = null;
            this.isInitialized = false;
            throw error; // Relanzar el error para que el llamador sepa que falló
        }
    }

    /**
     * Realiza OCR en la(s) fuente(s) de imagen de entrada.
     * @param {HTMLImageElement | HTMLCanvasElement | ImageData | cv.Mat | Array<HTMLImageElement | HTMLCanvasElement | ImageData | cv.Mat>} source - Imagen(es) de entrada.
     * @param {boolean} [returnConfidence=false] - Si se deben devolver las puntuaciones de confianza.
     * @returns {Promise<string[] | [string[], number[][]]>} Promesa que se resuelve con las matrículas decodificadas o [matrículas, confianzas].
     * @throws {Error} Si no está inicializado, OpenCV no está listo, o si falla la inferencia/procesamiento.
     */
    async run(source, returnConfidence = false) {
        if (!this.isInitialized || !this.session || !this.config) {
            throw new Error("Reconocedor no inicializado. Llama a initialize(modelName) primero.");
        }
        if (!window.cv) { // Re-verificar por si acaso
             throw new Error("OpenCV.js no está listo (window.cv no encontrado). No se pueden procesar imágenes.");
        }

        console.log("Iniciando ejecución OCR...");
        const startTime = performance.now();
        let inputMats = []; // Para asegurar limpieza en caso de error

        try {
            // 1. Cargar y Convertir Fuentes de Imagen a cv.Mat gris
            const loadStartTime = performance.now();
            inputMats = await _loadImageInputs(source); // Devuelve array de cv.Mat (escala de grises)
            // Dentro de OnnxOcrRecognizer.run, después de llamar a _loadImageInputs
            console.log(`Entrada para preprocesamiento: ${inputMats.length} cv.Mat(s)`);
            if (inputMats.length > 0) {
                // Loguear dimensiones del primer Mat como ejemplo
                const firstMat = inputMats[0];
                console.log(`Dimensiones del primer Mat (Alto x Ancho): ${firstMat.rows} x ${firstMat.cols}, Canales: ${firstMat.channels()}`);
                // OPCIONAL: Visualizar la imagen que entra a preprocesamiento (requiere un canvas en el HTML)
                // try {
                //     const tempCanvas = document.getElementById('debugCanvas'); // Necesitas <canvas id="debugCanvas"></canvas> en tu HTML
                //     if (tempCanvas) cv.imshow(tempCanvas, firstMat);
                // } catch(e) { console.warn("No se pudo mostrar imagen en debugCanvas", e); }
            }
            if (inputMats.length === 0){
                 console.warn("Ninguna imagen válida procesada desde la fuente. Devolviendo resultado vacío.");
                 return returnConfidence ? [[], []] : [];
            }
            console.log(`Carga y conversión de ${inputMats.length} imagen(es) completada en ${(performance.now() - loadStartTime).toFixed(2)} ms.`);

            // 2. Preprocesar Imágenes (Redimensionar, etc.)
            const prepStartTime = performance.now();
            const preprocessed = OCRUtils.preprocessImage(inputMats, this.config.img_height, this.config.img_width);
            console.log(`Preprocesamiento completado en ${(performance.now() - prepStartTime).toFixed(2)} ms. Forma de entrada: [${preprocessed.shape.join(', ')}]`);
            // --- AÑADIR ESTOS LOGS ---
            console.log(`Datos preprocesados - Forma esperada: [${preprocessed.shape.join(', ')}]`);
            // Muestra los primeros ~20 valores del array Uint8Array para ver si parecen píxeles (0-255)
            console.log(`Datos preprocesados - Muestra de datos (Uint8): [${preprocessed.data.slice(0, 20).join(', ')}]`);
            console.log(`Datos preprocesados - Longitud total: ${preprocessed.data.length}`);
            // Liberar memoria de los Mats creados/usados en el paso 1 AHORA
            console.debug(`Liberando ${inputMats.length} cv.Mat usados para preprocesamiento...`);
            inputMats.forEach((mat, index) => {
                // Solo borrar si tiene el método delete (evita errores si algo raro pasó)
                if(mat && typeof mat.delete === 'function') {
                     try { mat.delete(); } catch(e) { console.error(`Error al borrar mat ${index}:`, e); }
                }
            });
            inputMats = []; // Vaciar array para evitar doble borrado

            // 3. Ejecutar Inferencia ONNX
            const infStartTime = performance.now();
            // Asumir que el nombre de entrada es el primero (¡VERIFICA TU MODELO!)
            const inputName = this.session.inputNames[0];
            if (!inputName) throw new Error("No se pudo determinar el nombre de entrada del modelo.");

            const inputTensor = new ort.Tensor('uint8', preprocessed.data, preprocessed.shape); // <<< CAMBIAR a 'uint8'
            const feeds = { [inputName]: inputTensor };

            // Ejecutar la inferencia
            const results = await this.session.run(feeds);

            // Asumir que el nombre de salida es el primero (¡VERIFICA TU MODELO!)
            const outputName = this.session.outputNames[0];
            if (!outputName) throw new Error("No se pudo determinar el nombre de salida del modelo.");

            const outputTensor = results[outputName];
            if (!outputTensor) throw new Error(`Tensor de salida '${outputName}' no encontrado en los resultados.`);

            const outputData = outputTensor.data; // Float32Array (o el tipo que devuelva el modelo)
            const outputShape = outputTensor.dims; // Array de números [B, S, A] o [B, S*A]
            console.log(`Inferencia completada en ${(performance.now() - infStartTime).toFixed(2)} ms. Forma de salida: [${outputShape.join(', ')}]`);

            // 4. Postprocesar Salida del Modelo
            const postStartTime = performance.now();
            console.log("Entrada a postprocessOutput - Forma:", outputShape);
            console.log("Entrada a postprocessOutput - max_plate_slots:", this.config.max_plate_slots);
            console.log("Entrada a postprocessOutput - alphabet:", this.config.alphabet);
            console.log(`Salida cruda del modelo - Forma: [${outputShape.join(', ')}]`);
            // ¡¡ADVERTENCIA: Esto puede imprimir MUCHOS números!! Útil para depurar.
            // Si es demasiado, considera mostrar solo una parte o calcular algunos estadísticos (min, max, avg).
            console.log("Salida cruda del modelo - Datos:", outputData);
            const finalResult = OCRUtils.postprocessOutput(
                outputData,
                outputShape,
                this.config.max_plate_slots,
                this.config.alphabet,
                returnConfidence
            );
            console.log(`Postprocesamiento completado en ${(performance.now() - postStartTime).toFixed(2)} ms.`);

            console.log(`Ejecución OCR total completada en ${(performance.now() - startTime).toFixed(2)} ms.`);
            return finalResult;

        } catch (error) {
             console.error("Error durante la ejecución de OnnxOcrRecognizer.run:", error);
             // Asegurarse de limpiar Mats si hubo un error después de cargarlos pero antes de borrarlos
             if(inputMats && inputMats.length > 0) {
                 console.warn(`Intentando limpiar ${inputMats.length} Mats restantes después de un error...`);
                 inputMats.forEach((mat, index) => {
                    if(mat && typeof mat.delete === 'function') {
                         try { mat.delete(); } catch(e) { console.error(`Error al borrar mat ${index} durante limpieza de error:`, e); }
                    }
                });
             }
             throw error; // Relanzar para que el llamador maneje el fallo
        }
    }
}