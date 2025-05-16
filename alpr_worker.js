// Contenido REVISADO para alpr-worker.js

//self.importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js');
//self.importScripts('https://docs.opencv.org/4.10.0/opencv.js'); // Asegúrate que la versión coincida

import { ALPR } from './fastalprjs/src/alpr.js';

let alprInstance = null;
let cvReady = false;
let ortReady = false;

async function setupGlobals() {
    // Configurar ONNX Runtime
    try {
        if (typeof ort === 'undefined') {
            postMessage({ type: 'initError', error: 'ONNX Runtime (ort) no definido tras importScripts.' });
            return false;
        }
        // Usar self.ort para asegurar que estamos usando el del scope del worker
        self.ort.env.wasm.numThreads = Math.max(1, Math.min(navigator.hardwareConcurrency || 2, 2));
        self.ort.env.wasm.simd = true;
        self.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/'; // CDN path
        console.log(`ALPR Worker: ONNX Runtime configurado. Hilos: ${self.ort.env.wasm.numThreads}.`);
        ortReady = true;
    } catch (e) {
        console.error("ALPR Worker: Error configurando ONNX Runtime:", e);
        postMessage({ type: 'initError', error: `Error configurando ORT: ${e.message}` });
        return false;
    }

    // Configurar OpenCV.js
    // OpenCV.js cargado con importScripts usualmente define 'cv' globalmente.
    // El objeto 'cv' puede tener una propiedad 'onRuntimeInitialized' para cuando esté completamente listo.
    return new Promise((resolve) => {
        if (typeof cv !== 'undefined' && cv.Mat) { // Una comprobación básica de que cv existe
            if (cv.onRuntimeInitialized) { // Si existe el callback, usarlo
                 cv.onRuntimeInitialized = () => {
                    console.log("ALPR Worker: OpenCV.js Runtime Inicializado (vía callback).");
                    cvReady = true;
                    resolve(true);
                };
            } else { // Si no, asumir que está listo tras importScripts (común)
                console.log("ALPR Worker: OpenCV.js parece listo (sin callback onRuntimeInitialized).");
                cvReady = true;
                resolve(true);
            }
        } else {
            // Reintentar un poco por si tarda en aparecer cv en el scope global
            let retries = 0;
            const maxRetries = 50; // 5 segundos
            const cvPollInterval = setInterval(() => {
                if (typeof cv !== 'undefined' && cv.Mat) {
                    clearInterval(cvPollInterval);
                    console.log("ALPR Worker: OpenCV.js detectado tras polling.");
                     if (cv.onRuntimeInitialized) {
                        cv.onRuntimeInitialized = () => {
                            console.log("ALPR Worker: OpenCV.js Runtime Inicializado (vía callback tras polling).");
                            cvReady = true;
                            resolve(true);
                        };
                     } else {
                        cvReady = true;
                        resolve(true);
                     }
                } else if (++retries > maxRetries) {
                    clearInterval(cvPollInterval);
                    console.error("ALPR Worker: OpenCV.js (cv) no definido tras importScripts y polling.");
                    postMessage({ type: 'initError', error: 'OpenCV.js no se pudo cargar en el worker.' });
                    resolve(false);
                }
            }, 100);
        }
    });
}


async function initializeAlpr(detectorModel, ocrModelName) { // Renombrado ocrModel a ocrModelName
    console.log(`ALPR Worker: Iniciando ALPR. Detector: ${detectorModel}, OCR: ${ocrModelName}`);
    if (!ortReady || !cvReady) {
        const errorMsg = `ALPR no puede inicializar: ORT listo=${ortReady}, CV listo=${cvReady}`;
        console.error("ALPR Worker:", errorMsg);
        // No enviar postMessage aquí, ya se envió en setupGlobals si falló
        return false;
    }

    try {
        alprInstance = new ALPR({
            detectorModel: detectorModel,
            ocrModel: ocrModelName, // Este es el nombre/identificador del modelo OCR
        });

        // La carga de los modelos ONNX se dispara aquí o en el primer predict
        // Es mejor forzar la carga aquí para capturar errores temprano.
        // El constructor de DefaultDetector (en alpr.js) instancia LicensePlateDetector.
        // LicensePlateDetector (en openimageclaude.js) instancia YoloV9ObjectDetector.
        // El constructor de YoloV9ObjectDetector llama a this.loadModel().
        if (alprInstance && alprInstance.detector && alprInstance.detector.detector) { // YoloV9ObjectDetector
             if (!alprInstance.detector.detector.modelLoaded) { // Acceder a la propiedad de YoloV9ObjectDetector
                console.log("ALPR Worker: Cargando modelo de detector YOLOv9...");
                await alprInstance.detector.detector.loadModel(); // Este es async
            }
        } else {
            throw new Error("Instancia del detector (YoloV9ObjectDetector) no encontrada o no creada correctamente.");
        }

        // El constructor de DefaultOCR (en alpr.js) instancia OnnxOcrRecognizer.
        // OnnxOcrRecognizer (en fast-plate-ocr-js) tiene un método async initialize().
        if (alprInstance && alprInstance.ocr && alprInstance.ocr.ocrModel) { // OnnxOcrRecognizer
            if (!alprInstance.ocr.ocrModel.isInitialized) {
                console.log(`ALPR Worker: Inicializando modelo OCR (${ocrModelName})...`);
                await alprInstance.ocr.ocrModel.initialize(ocrModelName); // Pasar el nombre del modelo
            }
        } else {
            throw new Error("Instancia del reconocedor OCR (OnnxOcrRecognizer) no encontrada o no creada correctamente.");
        }

        console.log("ALPR Worker: Instancia ALPR y modelos subyacentes listos.");
        return true;
    } catch (error) {
        console.error("ALPR Worker: Error al inicializar instancia ALPR o cargar modelos:", error, error.stack);
        alprInstance = null;
        return false;
    }
}

self.onmessage = async function (e) {
    const { type, payload, frameId } = e.data;

    if (type === 'init') {
        const globalsOk = await setupGlobals();
        if (globalsOk) {
            const alprSuccess = await initializeAlpr(payload.detectorModel, payload.ocrModel);
            self.postMessage({ type: 'initComplete', success: alprSuccess, error: alprSuccess ? null : 'Fallo al inicializar ALPR en worker.' });
        } else {
            // El error ya fue enviado por setupGlobals
        }
    } else if (type === 'processFrame') {
        // ... (resto del código de processFrame como en la respuesta anterior, usando imageMat y borrándolo)
        if (!alprInstance || !cvReady || !ortReady) {
            self.postMessage({
                type: 'frameProcessed',
                frameId: frameId,
                error: `ALPR (listo=${!!alprInstance}), OpenCV (listo=${cvReady}) u ORT (listo=${ortReady}) no inicializado en worker.`,
                results: []
            });
            return;
        }

        const imageData = payload;
        let imageMat = null;
        try {
            // 'cv' debe estar disponible globalmente aquí
            imageMat = cv.matFromImageData(imageData);

            const alprResults = await alprInstance.predict(imageMat);

            self.postMessage({
                type: 'frameProcessed',
                frameId: frameId,
                results: alprResults,
            });
        } catch (error) {
            console.error(`ALPR Worker: Error procesando frame ${frameId}:`, error, error.stack);
            self.postMessage({ type: 'frameProcessed', frameId: frameId, error: error.message, results: [] });
        } finally {
            if (imageMat && !imageMat.isDeleted()) {
                imageMat.delete();
            }
        }
    }
};
console.log("ALPR Worker: Script evaluado. Esperando mensajes.");