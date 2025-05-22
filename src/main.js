// src/main.js

const modelSelect = document.getElementById('detectorModelSelect');
const confThreshInput = document.getElementById('confThreshInput');
const cameraButton = document.getElementById('cameraButton');
const loader = document.getElementById('loader');
const statusDiv = document.getElementById('status');
const videoElement = document.getElementById('videoFeed');
const processedCanvas = document.getElementById('processedCanvas');
const resultsArea = document.getElementById('resultsArea');
const canvasCtx = processedCanvas.getContext('2d');

let alprWorker = null;
let isWorkerInitialized = false;
let isCameraRunning = false;
let videoStream = null;
let animationFrameId = null;
let frameCounter = 0;
let processingFrameInWorker = false;
const TARGET_FPS_PROCESSING = 5; // Ajusta según necesites
let lastProcessTime = 0;
let appConfig = null;
let currentImageElement = null; // Para guardar referencia a la imagen para dibujar resultados

function updateStatus(message, isError = false) {
    statusDiv.textContent = message;
    statusDiv.style.color = isError ? 'red' : '#666';
    if (isError) console.error(message); else console.info(message);
}

function showLoader(show) {
    loader.style.display = show ? 'inline-block' : 'none';
    cameraButton.disabled = show;
    modelSelect.disabled = show;
    confThreshInput.disabled = show;
}

async function loadAppConfigAndSetup() {
    showLoader(true);
    updateStatus("Cargando configuración de la aplicación...");
    try {
        // Ajusta la ruta si app-config.json no está en la raíz del servidor
        // Si index.html está en src/ y app-config.json en src/, "./app-config.json" es correcto.
        const response = await fetch('./app-config.json');
        if (!response.ok) throw new Error(`Error HTTP ${response.status} cargando app-config.json desde ${response.url}`);
        appConfig = await response.json();
        console.log("Configuración de la aplicación cargada:", appConfig);

        detectorModelSelect.innerHTML = '';
        for (const key in appConfig.detectors) {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = appConfig.detectors[key].name;
            detectorModelSelect.appendChild(option);
        }
        detectorModelSelect.value = appConfig.defaultSettings.detectorKey;
        confThreshInput.value = appConfig.defaultSettings.confThresh;

        updateStatus("Configuración cargada. Listo para inicializar modelos.");
        showLoader(false);
        cameraButton.disabled = false;

    } catch (error) {
        updateStatus(`Error cargando app-config.json: ${error.message}`, true);
        showLoader(false);
        cameraButton.disabled = true;
    }
}

async function initializeOrReinitializeAlprWorker() {
    if (!appConfig) {
        updateStatus("Configuración de la aplicación no cargada. No se puede inicializar el worker.", true);
        return false; // Devolver false para indicar fallo
    }
    showLoader(true);
    const selectedDetectorKey = detectorModelSelect.value;
    const selectedOcrKey = appConfig.defaultSettings.ocrKey;
    const currentConfThresh = parseFloat(confThreshInput.value);

    updateStatus(`Inicializando ALPR Worker con detector: ${appConfig.detectors[selectedDetectorKey].name}...`);

    if (alprWorker) {
        console.log("Terminando worker ALPR anterior...");
        alprWorker.terminate();
        alprWorker = null;
    }
    isWorkerInitialized = false;
    // Ajusta la ruta si alpr_worker.js no está en la raíz del servidor
    // Si index.html está en src/ y alpr_worker.js en src/, "./alpr_worker.js" es correcto.
    alprWorker = new Worker('./alpr_worker.js', { type: 'module' });

    return new Promise((resolve, reject) => {
        alprWorker.onmessage = function (e) {
            const { type, error, results, frameId, payload } = e.data;

            if (type === 'initComplete') {
                showLoader(false);
                if (payload && payload.success) {
                    isWorkerInitialized = true;
                    updateStatus(`ALPR Worker inicializado (${appConfig.detectors[selectedDetectorKey].name}).`);
                    if (isCameraRunning && !animationFrameId) {
                        lastProcessTime = performance.now();
                        animationFrameId = requestAnimationFrame(videoProcessingLoop);
                    }
                    resolve(true); // Resuelve la promesa en éxito
                } else {
                    const errorMessage = payload.error || 'Error desconocido durante la inicialización del worker.';
                    updateStatus(`Error al inicializar ALPR Worker: ${errorMessage}`, true);
                    alprWorker = null; // Considerar no anularlo para reintentos si es apropiado
                    resolve(false); // Resuelve la promesa indicando fallo
                }
            } else if (type === 'frameProcessed') {
                processingFrameInWorker = false;
                if (error) {
                    console.error(`Error del Worker (frame ${frameId}):`, error);
                } else if (results) {
                    drawAlprResults(results, processedCanvas, canvasCtx, videoElement); // Usa videoElement como referencia
                    if (results.length > 0) {
                        let resultsTextContent = `Detecciones: ${results.length}\n\n`;
                        results.forEach((item, index) => {
                            const det = item.detection;
                            const ocr = item.ocr;
                            const detConf = det && typeof det.confidence === 'number' ? (det.confidence * 100).toFixed(1) : "N/A";
                            const ocrText = ocr && ocr.text ? ocr.text : "N/A";
                            const ocrConf = ocr && typeof ocr.confidence === 'number' ? (ocr.confidence * 100).toFixed(1) : "N/A";
                            resultsTextContent += `Placa ${index + 1}: ${ocrText} (OCR:${ocrConf}%) (Det:${detConf}%)\n`;
                        });
                        resultsArea.textContent = resultsTextContent.trim();
                    }
                }
            } else if (type === 'initError') {
                showLoader(false);
                updateStatus(`Error crítico inicializando Worker: ${error}`, true);
                alprWorker = null;
                resolve(false);
            }
        };

        alprWorker.onerror = function(err) {
            console.error("Error en ALPR Worker:", err.message, err);
            updateStatus(`Error fatal en ALPR Worker: ${err.message}`, true);
            showLoader(false);
            isWorkerInitialized = false;
            if (alprWorker) alprWorker.terminate();
            alprWorker = null;
            reject(err); // Rechaza la promesa en error fatal del worker
        };

        const workerConfigPayload = {
            appConfig: appConfig,
            selectedDetectorKey: selectedDetectorKey,
            selectedOcrKey: selectedOcrKey,
            confThresh: currentConfThresh
        };
        console.log("Hilo Principal: Enviando mensaje 'INIT' al worker con payload:", workerConfigPayload);
        alprWorker.postMessage({ type: 'INIT', payload: workerConfigPayload });
    });
}

function drawAlprResults(alprResults, canvas, ctx, videoSourceElement) {
    if (!videoSourceElement || videoSourceElement.videoWidth === 0 || videoSourceElement.videoHeight === 0) {
        return;
    }
    // Sincronizar dimensiones del canvas de dibujo con las dimensiones *reales* del video
    if (canvas.width !== videoSourceElement.videoWidth || canvas.height !== videoSourceElement.videoHeight) {
        canvas.width = videoSourceElement.videoWidth;
        canvas.height = videoSourceElement.videoHeight;
    }

    // Ajustar el tamaño CSS del canvas para que coincida con el tamaño renderizado del elemento de video
    const videoRect = videoSourceElement.getBoundingClientRect();
    canvas.style.position = 'absolute'; // Asegurar que está posicionado sobre el video-container
    canvas.style.left = videoSourceElement.offsetLeft + 'px';
    canvas.style.top = videoSourceElement.offsetTop + 'px';
    canvas.style.width = videoRect.width + 'px';   // Tamaño CSS
    canvas.style.height = videoRect.height + 'px'; // Tamaño CSS

    ctx.clearRect(0, 0, canvas.width, canvas.height); // Limpiar con dimensiones de dibujo
    if (!alprResults || alprResults.length === 0) return;

    // Escalar coordenadas del BBox (que son relativas a videoWidth/videoHeight originales)
    // al tamaño actual del canvas de dibujo (que es videoWidth/videoHeight).
    // En este caso, si canvas.width = videoSourceElement.videoWidth, scaleX es 1.
    // Pero es bueno mantenerlo por si el canvas de dibujo tuviera un tamaño diferente al original del video.
    const scaleX = canvas.width / videoSourceElement.videoWidth;
    const scaleY = canvas.height / videoSourceElement.videoHeight;

    alprResults.forEach(item => {
        const { detection, ocr } = item;
        if (!detection || !detection.boundingBox) {
            console.warn("drawAlprResults: Item sin detección o boundingBox", item);
            return;
        }
        const bb = detection.boundingBox; // {x1, y1, x2, y2, width, height}

        // Las coordenadas bb.x1, bb.y1, bb.width, bb.height ya son relativas
        // a la imagen original que procesó el modelo.
        const x1 = bb.x1 * scaleX;
        const y1 = bb.y1 * scaleY;
        const width = bb.width * scaleX;
        const height = bb.height * scaleY;

        if (width <= 0 || height <= 0) {
            console.warn("drawAlprResults: Bounding box con ancho o alto inválido:", {x1, y1, width, height});
            return;
        }

        ctx.strokeStyle = 'lime';
        ctx.lineWidth = Math.max(1, Math.min(3, canvas.width / 300)); // Ajustar grosor
        ctx.strokeRect(x1, y1, width, height);

        const detConf = typeof detection.confidence === 'number' ? (detection.confidence * 100).toFixed(0) : "";
        const ocrTextLabel = ocr && ocr.text ? ocr.text : "";
        const ocrConfLabel = ocr && typeof ocr.confidence === 'number' ? (ocr.confidence * 100).toFixed(0) : "";
        const label = `${ocrTextLabel} (D:${detConf}% O:${ocrConfLabel}%)`;
        const fontHeight = Math.max(10, Math.min(16, Math.round(canvas.height / 30))); // Ajustar tamaño de fuente
        ctx.font = `bold ${fontHeight}px Arial`;
        ctx.textBaseline = 'bottom';

        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textBgHeight = fontHeight + 4;
        const textY = y1 - 2; // Posición del texto encima del bounding box

        ctx.fillStyle = 'rgba(0, 255, 0, 0.75)';
        ctx.fillRect(x1, textY - textBgHeight +2, textWidth + 4, textBgHeight); // Ajustar y del fondo
        ctx.fillStyle = 'black';
        ctx.fillText(label, x1 + 2, textY);
    });
}


async function startVideoStream() {
    if (isCameraRunning) return;
    if (!isWorkerInitialized) {
        updateStatus("Worker no inicializado. Por favor, selecciona un modelo y espera.", true);
        return;
    }
    updateStatus("Iniciando cámara...");
    showLoader(true);
    try {
        const constraints = { video: { facingMode: "environment" }, audio: false };
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = videoStream;
        videoElement.onloadedmetadata = () => {
            // No es necesario ajustar processedCanvas.width/height aquí si se hace en drawAlprResults
            // y al inicio de videoProcessingLoop si es necesario para la captura.
            isCameraRunning = true;
            cameraButton.textContent = 'Detener Cámara';
            cameraButton.classList.add('stop');
            showLoader(false);
            updateStatus("Cámara iniciada. Procesando...");
            lastProcessTime = performance.now();
            processingFrameInWorker = false;
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
            animationFrameId = requestAnimationFrame(videoProcessingLoop);
        };
    } catch (error) {
        updateStatus(`Error al acceder a la cámara: ${error.message}`, true);
        isCameraRunning = false;
        showLoader(false);
    }
}

function stopVideoStream() {
    if (!isCameraRunning) return;
    updateStatus("Deteniendo cámara...");
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }
    videoElement.srcObject = null;
    videoElement.pause(); // Asegurar que el video se pause
    isCameraRunning = false;
    cameraButton.textContent = 'Iniciar Cámara';
    cameraButton.classList.remove('stop');
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    canvasCtx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
    resultsArea.textContent = '';
    updateStatus("Cámara detenida.");
    showLoader(false);
    processingFrameInWorker = false;
}

function videoProcessingLoop(currentTime) {
    if (!isCameraRunning) {
        if (animationFrameId) cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
        return;
    }

    const deltaTime = currentTime - lastProcessTime;
    const interval = 1000 / TARGET_FPS_PROCESSING;

    if (deltaTime >= interval && !processingFrameInWorker && isWorkerInitialized &&
        videoElement.readyState >= videoElement.HAVE_METADATA && videoElement.videoWidth > 0) {
        lastProcessTime = currentTime - (deltaTime % interval); // Corregir para mantener el ritmo
        processingFrameInWorker = true;
        frameCounter++;

        const tempCaptureCanvas = document.createElement('canvas');
        tempCaptureCanvas.width = videoElement.videoWidth;
        tempCaptureCanvas.height = videoElement.videoHeight;
        const tempCtx = tempCaptureCanvas.getContext('2d', { willReadFrequently: true }); // willReadFrequently puede ayudar
        tempCtx.drawImage(videoElement, 0, 0, tempCaptureCanvas.width, tempCaptureCanvas.height);
        try {
            const imageData = tempCtx.getImageData(0, 0, tempCaptureCanvas.width, tempCaptureCanvas.height);
            alprWorker.postMessage({ type: 'processFrame', payload: imageData, frameId: frameCounter }, [imageData.data.buffer]);
        } catch(e) {
            console.error("Error obteniendo ImageData del video:", e);
            processingFrameInWorker = false; // Liberar flag si falla la captura
        }

    }
    animationFrameId = requestAnimationFrame(videoProcessingLoop);
}

// Event Listeners
modelSelect.addEventListener('change', async () => {
    if (isCameraRunning) stopVideoStream();
    await initializeOrReinitializeAlprWorker();
});
confThreshInput.addEventListener('change', async () => {
    if (isCameraRunning) stopVideoStream();
    await initializeOrReinitializeAlprWorker();
});

cameraButton.addEventListener('click', async () => {
    if (!appConfig) {
        await loadAppConfigAndSetup();
        if (!appConfig) return;
    }
    if (isCameraRunning) {
        stopVideoStream();
    } else {
        // Si el worker no está listo, initializeOrReinitializeAlprWorker lo intentará.
        // Y si tiene éxito, startVideoStream será llamado.
        const workerReady = await initializeOrReinitializeAlprWorker();
        if (workerReady) { // Solo iniciar si el worker se inicializó bien
            startVideoStream();
        } else {
            updateStatus("Fallo al inicializar el worker. No se puede iniciar la cámara.", true);
        }
    }
});

// Inicialización al Cargar la Página
document.addEventListener('DOMContentLoaded', () => {
    cameraButton.disabled = true; // Deshabilitar hasta que la config se cargue
    loadAppConfigAndSetup();
});