// alpr_worker.js

// Importar ONNX Runtime y ALPR
import * as ortNS from './libs/onnxruntime-web/dist/ort.mjs'; // Ajusta la ruta si es necesario
self.ort = ortNS; // Hacer ONNX Runtime disponible globalmente como self.ort

import ALPR from './alpr/src/alpr.js'; // ALPR debe ser la clase exportada por defecto

let alprInstance = null;
let cvReady = false; // Se volverá true cuando OpenCV esté completamente listo
let ortReady = false;
let currentAppConfig = null;

/**
 * Carga y espera la inicialización completa de OpenCV.js.
 * Retorna una promesa que se resuelve cuando OpenCV está listo, o se rechaza si falla.
 */
async function loadAndInitializeOpenCV() {
    return new Promise(async (resolve, reject) => {
        if (cvReady) { // Si por alguna razón ya está listo
            console.info("ALPR Worker: OpenCV ya estaba marcado como listo.");
            resolve(true);
            return;
        }

        // Comprobar si 'cv' ya existe y está funcional (ej. si el worker se reutiliza y ya cargó)
        if (typeof cv !== 'undefined' && cv.Mat) {
            console.info("ALPR Worker: OpenCV (cv.Mat) parece estar ya disponible y funcional.");
            cvReady = true;
            resolve(true);
            return;
        }

        console.info("ALPR Worker: Intentando cargar y ejecutar opencv.js...");
        try {
            const response = await fetch('./libs/opencv-js/dist/opencv.js'); // Asegúrate que la ruta sea correcta
            if (!response.ok) {
                throw new Error(`Error HTTP ${response.status} cargando OpenCV.js desde ${response.url}`);
            }
            const scriptText = await response.text();
            // Ejecutar el script de OpenCV en el contexto global del worker
            // Es crucial que el script de OpenCV se ejecute correctamente aquí
            // y defina el objeto 'cv' global y su callback 'onRuntimeInitialized'.
            new Function(scriptText).call(self);

            if (typeof cv === 'undefined') {
                throw new Error("'cv' no está definido globalmente después de ejecutar el script de opencv.js. Verifica el contenido y la carga del script.");
            }
            console.info("ALPR Worker: opencv.js script ejecutado. Esperando inicialización del runtime WASM...");

            // Esperar a que el runtime de OpenCV (WASM) esté completamente listo
            if (typeof cv.onRuntimeInitialized === 'function') {
                cv.onRuntimeInitialized = () => {
                    if (cvReady) return; // Prevenir múltiples resoluciones
                    console.info("ALPR Worker: OpenCV.js Runtime Inicializado (vía callback cv.onRuntimeInitialized).");
                    cvReady = true;
                    resolve(true);
                };
            } else {
                // Fallback: Si onRuntimeInitialized no es una función (versiones antiguas/diferentes de OpenCV.js)
                // Intentar un sondeo para cv.Mat como indicador de que está listo.
                console.warn("ALPR Worker: cv.onRuntimeInitialized no es una función. Iniciando sondeo para cv.Mat...");
                let retries = 0;
                const maxRetries = 50; // Aumentar intentos (ej. 5 segundos si el intervalo es 100ms)
                const intervalId = setInterval(() => {
                    if (typeof cv !== 'undefined' && cv.Mat) {
                        clearInterval(intervalId);
                        if (cvReady) return; // Prevenir múltiples resoluciones
                        console.info("ALPR Worker: OpenCV.js (cv.Mat) detectado como disponible tras sondeo.");
                        cvReady = true;
                        resolve(true);
                    } else if (++retries > maxRetries) {
                        clearInterval(intervalId);
                        const errorMsg = 'OpenCV.js (cv.Mat) no disponible después de sondeo prolongado. El runtime WASM podría no haberse cargado o inicializado.';
                        console.error("ALPR Worker:", errorMsg);
                        reject(new Error(errorMsg));
                    }
                }, 100);
            }
        } catch (error) {
            console.error("ALPR Worker: Fallo crítico durante la carga o inicialización de OpenCV.js:", error);
            postMessage({ type: 'initError', error: `Fallo crítico en OpenCV: ${error.message || String(error)}` });
            reject(error);
        }
    });
}


async function setupOrtEnvironment() {
    try {
        if (typeof self.ort === 'undefined' || typeof self.ort.InferenceSession === 'undefined') {
            throw new Error('ONNX Runtime (self.ort) no disponible o no es un módulo válido.');
        }
        self.ort.env.wasm.numThreads = Math.max(1, Math.min(navigator.hardwareConcurrency || 2, 4));
        self.ort.env.wasm.simd = true;


        console.info(`ALPR Worker: Entorno WASM de ONNX Runtime configurado. Hilos: ${self.ort.env.wasm.numThreads}, SIMD: ${self.ort.env.wasm.simd}. WasmPaths: '${self.ort.env.wasm.wasmPaths}'`);
        ortReady = true;
        return true;
    } catch (e) {
        console.error("ALPR Worker: Error configurando entorno de ONNX Runtime:", e);
        postMessage({ type: 'initError', error: `Error ORT env: ${e.message}` });
        return false;
    }
}

// La función initializeAlprInstance permanece igual que en tu última versión.
// Solo asegúrate que verifique cvReady y ortReady.
async function initializeAlprInstance(workerInitPayload) {
    currentAppConfig = workerInitPayload.appConfig;
    const { selectedDetectorKey, selectedOcrKey, confThresh } = workerInitPayload;

    if (!ortReady || !cvReady) { // Estas flags ahora son más confiables
        const errorMsg = `ALPR no puede inicializar: ORT listo=${ortReady}, CV listo=${cvReady}`;
        console.error("ALPR Worker:", errorMsg);
        postMessage({ type: 'initComplete', payload: { success: false, error: errorMsg } });
        return; // No continuar
    }
    if (!currentAppConfig) {
        postMessage({ type: 'initComplete', payload: { success: false, error: "Configuración no recibida." } });
        return;
    }

    try {
        const detectorSettings = currentAppConfig.detectors[selectedDetectorKey];
        const ocrSettings = currentAppConfig.ocrModels[selectedOcrKey];
        const commonSettings = currentAppConfig.commonParameters;
        const pathSettings = currentAppConfig.paths;

        if (!detectorSettings) throw new Error(`Config para detector '${selectedDetectorKey}' no encontrada.`);
        if (!ocrSettings) throw new Error(`Config para OCR '${selectedOcrKey}' no encontrada.`);

        // Rutas: estas deben ser relativas a la ubicación del worker o absolutas en el servidor.
        // Si app-config.json usa "./models/" y index.html está en la raíz del servidor (o en src/ y el servidor sirve src/ como raíz),
        // y el worker está en `src/alpr_worker.js` (servido como `/alpr_worker.js` o `/src/alpr_worker.js`),
        // entonces "./models/" desde la perspectiva del fetch en el worker será relativo a la ubicación del worker.
        // Es más seguro si las rutas en app-config.json son rutas absolutas desde la raíz del servidor
        // o si construyes rutas relativas al worker aquí.
        // Ejemplo: si appConfig.paths.baseModelsPath es "./models/" (relativo a index.html/raíz del servidor)
        // y el worker está en `src/alpr_worker.js`, la ruta desde el worker al modelo sería `../models/`.
        // Vamos a asumir que las rutas en `app-config.json` (ej. "./models/") son relativas a la raíz del servidor.

        const baseModelsPath = pathSettings.baseModelsPath; // ej: "./models/" (relativo a la raíz del servidor)

        const detectorModelsFullPath = `${baseModelsPath}${pathSettings.detectorModelsSubPath}`; // ej: "./models/detection/"

        let ocrModelFileFullPath = null;
        if (ocrSettings.modelFileName) {
            ocrModelFileFullPath = `${baseModelsPath}${pathSettings.ocrModelsSubPath}${ocrSettings.modelFileName}`;
        }
        let ocrConfigFullPath = null;
        if (ocrSettings.configFileName) {
             ocrConfigFullPath = pathSettings.ocrConfigsSubPath ?
                `${baseModelsPath}${pathSettings.ocrConfigsSubPath}${ocrSettings.configFileName}` :
                `${baseModelsPath}${ocrSettings.configFileName}`; // Si ocrConfigsSubPath es ""
             // Si tu global_mobile_vit_v2_ocr_config.json está en src/ y no en models/ocr/configs/
             // entonces app-config.json debería reflejarlo, o ajusta la ruta aquí.
             // Por ejemplo, si configFileName es "global_mobile_vit_v2_ocr_config.json"
             // y está en la raíz del proyecto servido: ocrConfigFullPath = `./${ocrSettings.configFileName}`;
             console.log("Ruta config OCR construida a pasar a ALPR:", ocrConfigFullPath);
        }


        alprInstance = new ALPR({
            detectorModel: detectorSettings.fileName,
            detectorConfThresh: confThresh,
            detectorModelsPath: detectorModelsFullPath, // Ruta a la carpeta de modelos de detección
            heightInput: detectorSettings.inputHeight,
            widthInput: detectorSettings.inputWidth,
            detectorExecutionProviders: commonSettings.onnxExecutionProviders,

            ocrModel: ocrSettings.hubName || ocrSettings.name,
            ocrModelPath: ocrModelFileFullPath,
            ocrConfigPath: ocrConfigFullPath,
            ocrForceDownload: commonSettings.ocrForceDownload,
            ocrExecutionProviders: commonSettings.onnxExecutionProviders
        });

        await alprInstance.init();
        console.info("ALPR Worker: Instancia ALPR y modelos inicializados.");
        postMessage({ type: 'initComplete', payload: { success: true } });

    } catch (error) {
        console.error("ALPR Worker: Error inicializando ALPR con config:", error, error.stack);
        alprInstance = null;
        postMessage({ type: 'initComplete', payload: { success: false, error: error.message } });
    }
}


self.onmessage = async function (e) {
    const { type, payload, frameId } = e.data;

    switch (type) {
        case 'INIT':
            console.log("ALPR Worker: Recibido INIT, payload:", payload);
            currentAppConfig = payload.appConfig;

            try {
                const ortEnvOk = await setupOrtEnvironment();
                if (!ortEnvOk) {
                    // El mensaje de error ya se envió desde setupOrtEnvironment
                    postMessage({ type: 'initComplete', payload: { success: false, error: 'Fallo configurando entorno ORT (detalle en log anterior).' } });
                    return;
                }

                // Cargar e inicializar OpenCV. Esta función ahora maneja la espera y errores.
                await loadAndInitializeOpenCV(); // Esto establece cvReady si tiene éxito, o rechaza.

                // Si llegamos aquí y cvReady es true (y ortReady es true), entonces inicializar ALPR.
                if (ortReady && cvReady) {
                    await initializeAlprInstance(payload);
                } else {
                    // Si cvReady no es true, loadAndInitializeOpenCV debe haber rechazado y enviado un mensaje.
                    // Este es un fallback.
                    const libError = !ortReady ? 'ORT no listo.' : (!cvReady ? 'OpenCV no listo (después del intento de carga).' : 'Bibliotecas desconocidas no listas.');
                    console.error("ALPR Worker: No se pudo proceder a initializeAlprInstance.", libError);
                    postMessage({ type: 'initComplete', payload: { success: false, error: `Fallo en la carga de bibliotecas: ${libError}` } });
                }
            } catch (initProcessError) {
                // Capturar errores de loadAndInitializeOpenCV si la promesa es rechazada
                console.error("ALPR Worker: Error durante la secuencia INIT (posiblemente de OpenCV):", initProcessError);
                postMessage({ type: 'initComplete', payload: { success: false, error: `Error general en INIT: ${initProcessError.message || String(initProcessError)}` } });
            }
            break;

        case 'processFrame':
            if (!alprInstance || !alprInstance.isInitialized) {
                self.postMessage({ type: 'frameProcessed', frameId: frameId, error: `Worker no listo: ALPR init=${alprInstance?.isInitialized}`, results: [] });
                return;
            }
            const imageData = payload;
            let imageMat = null;
            try {
                if (!cv || !cv.matFromImageData) throw new Error("cv.matFromImageData no disponible.");
                imageMat = cv.matFromImageData(imageData);
                const startTime = performance.now();
                const alprResults = await alprInstance.predict(imageMat);
                const endTime = performance.now();
                self.postMessage({ type: 'frameProcessed', frameId: frameId, results: alprResults, time: endTime - startTime });
            } catch (error) {
                console.error(`ALPR Worker: Error procesando frame ${frameId}:`, error, error.stack);
                self.postMessage({ type: 'frameProcessed', frameId: frameId, error: error.message, results: [] });
            } finally {
                if (imageMat && !imageMat.isDeleted()) imageMat.delete();
            }
            break;
        default:
            console.warn("ALPR Worker: Mensaje tipo desconocido:", type);
    }
};
console.info("ALPR Worker: Script evaluado. Esperando mensajes.");