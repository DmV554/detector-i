// alpr_worker.js

// 1. Importar ONNX Runtime como un módulo ES6 desde la carpeta libs
// Asegúrate que la ruta './libs/onnxruntime-web/dist/ort.mjs' sea correcta.
// Podría ser ort.min.mjs, ort.all.mjs, etc., dependiendo del archivo específico que uses.
import * as ortNS from './libs/onnxruntime-web/dist/ort.mjs';
self.ort = ortNS; // Hacer ONNX Runtime disponible globalmente como self.ort

// Variables para el estado de carga y la instancia de ALPR
let ALPR; // Se importará dinámicamente
let alprInstance = null;
let cvReady = false;
let ortReady = false;
let openCVLoadSuccess = false;

// 2. Función para cargar OpenCV.js
// Se llamará antes de cualquier inicialización que dependa de OpenCV.
async function loadOpenCV() {
  try {
    const response = await fetch('./libs/opencv-js/dist/opencv.js'); // Carga el script
    if (!response.ok) {
      throw new Error(`Error HTTP cargando OpenCV! status: ${response.status}`);
    }
    const scriptText = await response.text();
    // Ejecuta el script en el ámbito global del worker.
    // new Function(scriptText).call(self) intenta asegurar que 'this' dentro del script sea 'self'.
    new Function(scriptText).call(self);

    // Verifica si 'cv' se definió globalmente. Emscripten (OpenCV.js) usualmente lo hace.
    if (typeof cv === 'undefined') {
      throw new Error("'cv' no está definido globalmente después de ejecutar opencv.js. Verifica el archivo opencv.js.");
    }
    console.info("ALPR Worker: OpenCV.js obtenido y ejecutado localmente.");
    return true; // Indica éxito
  } catch (e) {
    console.error("ALPR Worker: Fallo crítico al cargar o ejecutar opencv.js local.", e);
    // Envía un mensaje de error al hilo principal.
    postMessage({ type: 'initError', error: `Fallo al cargar/ejecutar opencv.js local: ${e.message || String(e)}` });
    return false; // Indica fallo
  }
}

// 3. Función para configurar las variables globales y esperar a que las bibliotecas estén listas
async function setupGlobals() {
    // Configurar ONNX Runtime (ya debería estar disponible como self.ort)
    try {
        if (typeof self.ort === 'undefined' || typeof self.ort.InferenceSession === 'undefined') {
            postMessage({ type: 'initError', error: 'ONNX Runtime (self.ort) no inicializado correctamente.' });
            return false;
        }
        self.ort.env.wasm.numThreads = Math.max(1, Math.min(navigator.hardwareConcurrency || 2, 2));
        self.ort.env.wasm.simd = true; // Habilitar SIMD si tus archivos wasm lo soportan (ej. ort-wasm-simd.wasm)

        // === CAMBIO CRÍTICO ===
        // Elimina o comenta la siguiente línea para permitir que ONNX Runtime
        // use su comportamiento predeterminado para encontrar los archivos .wasm.
        // Por defecto, los busca en el mismo directorio que el archivo ort.mjs principal.
        // self.ort.env.wasm.wasmPaths = './'; // Comentar o eliminar esta línea

        // Si necesitas explícitamente que busque en la carpeta dist (donde está ort.mjs y sus hermanos .wasm)
        // y tu worker está en la raíz, la ruta correcta sería:
        // self.ort.env.wasm.wasmPaths = './libs/onnxruntime-web/dist/';
        // PERO, para probar el default, primero comenta/elimina cualquier asignación a wasmPaths.
        // Si el default no funciona, la línea anterior podría ser la correcta.

        console.info(`ALPR Worker: ONNX Runtime configurado. Hilos: ${self.ort.env.wasm.numThreads}, SIMD: ${self.ort.env.wasm.simd}. Wasm Paths: (usando predeterminado o ruta específica).`);
        ortReady = true;
    } catch (e) {
        console.error("ALPR Worker: Error configurando ONNX Runtime:", e);
        postMessage({ type: 'initError', error: `Error configurando ORT: ${e.message}` });
        return false;
    }

    // Configurar OpenCV.js (esto permanece igual que en la respuesta anterior)
    if (!openCVLoadSuccess) {
        postMessage({ type: 'initError', error: 'OpenCV no se cargó previamente con éxito.' });
        return false;
    }

    return new Promise((resolve) => {
        if (typeof cv !== 'undefined' && cv.Mat) {
            if (cv.onRuntimeInitialized && typeof cv.onRuntimeInitialized === 'function') {
                cv.onRuntimeInitialized = () => {
                    console.info("ALPR Worker: OpenCV.js Runtime Inicializado (vía callback).");
                    cvReady = true;
                    resolve(true);
                };
            } else {
                console.info("ALPR Worker: OpenCV.js parece listo (sin callback onRuntimeInitialized o no es una función).");
                cvReady = true;
                resolve(true);
            }
        } else {
            let retries = 0;
            const maxRetries = 20;
            const cvPollInterval = setInterval(() => {
                if (typeof cv !== 'undefined' && cv.Mat) {
                    clearInterval(cvPollInterval);
                    console.info("ALPR Worker: OpenCV.js (cv.Mat) detectado tras sondeo en setupGlobals.");
                    if (cv.onRuntimeInitialized && typeof cv.onRuntimeInitialized === 'function') {
                         cv.onRuntimeInitialized = () => {
                            console.info("ALPR Worker: OpenCV.js Runtime Inicializado (vía callback después de sondeo en setupGlobals).");
                            cvReady = true;
                            resolve(true);
                        };
                    } else {
                        cvReady = true;
                        resolve(true);
                    }
                } else if (++retries > maxRetries) {
                    clearInterval(cvPollInterval);
                    console.error("ALPR Worker: OpenCV.js (cv.Mat) no disponible tras sondeo en setupGlobals.");
                    postMessage({ type: 'initError', error: 'OpenCV.js (cv.Mat) no disponible después del sondeo.' });
                    resolve(false);
                }
            }, 100);
        }
    });
}

// 4. Función para inicializar la instancia de ALPR
async function initializeAlpr(detectorModel, ocrModelName) {
    console.info(`ALPR Worker: Iniciando ALPR. Detector: ${detectorModel}, OCR: ${ocrModelName}`);
    if (!ortReady || !cvReady) {
        const errorMsg = `ALPR no puede inicializar: ORT listo=${ortReady}, CV listo=${cvReady}`;
        console.error("ALPR Worker:", errorMsg);
        return false;
    }
    if (!ALPR) { // Asegurarse que el módulo ALPR fue cargado
        console.error("ALPR Worker: El módulo ALPR no está cargado.");
        return false;
    }

    try {
        alprInstance = new ALPR({ // ALPR se importa dinámicamente
            detectorModelPath: "./models/",
            detectorModel: detectorModel,
            ocrModel: ocrModelName,
        });

        // Carga de los modelos internos de ONNX (detector y OCR)
        if (alprInstance && alprInstance.detector && alprInstance.detector.detector) {
            if (!alprInstance.detector.detector.modelLoaded) {
                console.info("ALPR Worker: Cargando modelo de detector...");
                await alprInstance.detector.detector.loadModel();
            }
        } else {
            throw new Error("Instancia del detector (YoloV9ObjectDetector) no encontrada en ALPR.");
        }

        if (alprInstance && alprInstance.ocr && alprInstance.ocr.ocrModel) {
            if (!alprInstance.ocr.ocrModel.isInitialized) {
                console.info(`ALPR Worker: Inicializando modelo OCR (${ocrModelName})...`);
                await alprInstance.ocr.ocrModel.initialize(ocrModelName);
            }
        } else {
            throw new Error("Instancia del reconocedor OCR (OnnxOcrRecognizer) no encontrada en ALPR.");
        }

        console.info("ALPR Worker: Instancia ALPR y modelos subyacentes listos.");
        return true;
    } catch (error) {
        console.error("ALPR Worker: Error al inicializar instancia ALPR o sus modelos:", error, error.stack);
        alprInstance = null;
        return false;
    }
}

// 5. Manejador de mensajes del hilo principal
self.onmessage = async function (e) {
    const { type, payload, frameId } = e.data;

    if (type === 'init') {
        // Carga OpenCV primero, ya que es la dependencia más problemática
        if (!openCVLoadSuccess) { // Solo intentar una vez o si falló antes
            openCVLoadSuccess = await loadOpenCV();
        }

        if (!openCVLoadSuccess) {
            // El mensaje de error ya fue enviado por loadOpenCV()
            self.postMessage({ type: 'initComplete', success: false, error: 'Fallo crítico al cargar OpenCV (reportado desde onmessage).' });
            return; // No continuar si OpenCV falló
        }

        // Importar dinámicamente ALPR *después* de que OpenCV se haya cargado (o intentado cargar)
        // Esto es crucial porque ALPR o sus dependencias (fast-plate-ocr-js, etc.)
        // pueden intentar usar 'cv' tan pronto como se evalúa el módulo.
        if (!ALPR) { // Solo importar si aún no se ha hecho
            try {
                const alprModule = await import('./alpr/src/alpr.js');
                ALPR = alprModule.ALPR; // Asigna la clase ALPR exportada
                if (!ALPR) {
                    throw new Error("La clase ALPR no se encontró en el módulo importado 'fastalprjs/src/alpr.js'.");
                }
            } catch (moduleError) {
                console.error("ALPR Worker: Fallo al importar dinámicamente ALPR.js.", moduleError);
                postMessage({ type: 'initError', error: `Fallo al importar ALPR.js: ${moduleError.message}` });
                self.postMessage({ type: 'initComplete', success: false, error: 'Fallo al importar el módulo ALPR.js.' });
                return;
            }
        }

        // Ahora, configurar globales (ORT y esperar inicialización de CV)
        const globalsOk = await setupGlobals();
        if (globalsOk) {
            const alprSuccess = await initializeAlpr(payload.detectorModel, payload.ocrModel);
            self.postMessage({ type: 'initComplete', success: alprSuccess, error: alprSuccess ? null : 'Fallo al inicializar ALPR en worker (después de setupGlobals).' });
        } else {
            self.postMessage({ type: 'initComplete', success: false, error: 'Fallo en setupGlobals (ORT o CV no listos/inicializados).' });
        }

    } else if (type === 'processFrame') {
        if (!alprInstance || !cvReady || !ortReady) {
            self.postMessage({
                type: 'frameProcessed',
                frameId: frameId,
                error: `Worker no está listo para procesar: ALPR instancia=${!!alprInstance}, CV listo=${cvReady}, ORT listo=${ortReady}.`,
                results: []
            });
            return;
        }

        const imageData = payload; // Es un objeto ImageData
        let imageMat = null;
        try {
            imageMat = cv.matFromImageData(imageData); // Convertir ImageData a cv.Mat
            const alprResults = await alprInstance.predict(imageMat); // predict espera cv.Mat

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
                imageMat.delete(); // Liberar memoria de la Mat de OpenCV
            }
        }
    }
};

console.info("ALPR Worker: Script evaluado. Esperando mensajes.");