import { ALPR } from './fastalprjs/src/alpr.js'; // Asegúrate que esta ruta es correcta

// Las configuraciones de ONNX Runtime Web se mantienen
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
const canUseThreads = self.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined';

if (canUseThreads) {
  ort.env.wasm.numThreads = Math.min(4, Math.max(1, navigator.hardwareConcurrency - 1));
}
ort.env.wasm.simd = true;

window.addEventListener("load", function () {
    let video, canvas, ctx;
    // let detectorSession, ocrSession; // Ya no se usan directamente
    let alprInstance; // Nueva instancia para ALPR
    let cameraStream = null;
    let animationId = null;
    let detectionRunning = false;
    let verificadorRunning = false;
    let drawLoopRunning = false;

    const gif_carga_principal = document.querySelector('.gif_carga_principal');
    const container = document.getElementById("detectionContainer");
    let mispatentes = [];

    var audioGris = new Audio('./z/sonido-gris.mp3');
    var audioRojo = new Audio('./z/sonido-rojo.mp3');

    function sonido_gris() {
        audioGris.play().catch(error => {
            console.error('Error al reproducir sonido Gris:', error);
        });
    }

    function sonido_rojo() {
        audioRojo.play().catch(error => {
            console.error('Error al reproducir sonido Rojo:', error);
        });
    }

    let expandido = false;
    const superCccCam = document.getElementById('super-ccc-cam');
    const toggleExpandirBtn = document.getElementById('toggleExpandirBtn');
    let imgExpandir = toggleExpandirBtn.querySelector('img');
    toggleExpandirBtn.addEventListener('click', toggleExpandir);
    function toggleExpandir() {
        if (!expandido) {
            imgExpandir.src = "./imagenes/iconos/expandirB.svg";
            superCccCam.className = "super-ccc-cam_expan padding_A";
            expandido = true;
        } else {
            imgExpandir.src = "./imagenes/iconos/expandirA.svg";
            superCccCam.className = "super-ccc-cam base_float";
            expandido = false;
        }
    }

    let VideoOculto = false;
    let ccc_cam = document.getElementById('ccc_cam');
    const toggleCamBtn = document.getElementById('toggleCamBtn');
    let imgVideoOculto = toggleCamBtn.querySelector('img');
    toggleCamBtn.addEventListener('click', toggleOcultar);
    function toggleOcultar() {
        if (!VideoOculto) {
            imgVideoOculto.src = "./imagenes/iconos/ocultarCamB.svg";
            ccc_cam.style.display = "none";
            VideoOculto = true;
        } else {
            imgVideoOculto.src = "./imagenes/iconos/ocultarCamA.svg";
            ccc_cam.style.display = "block";
            VideoOculto = false;
        }
    }

    const toggleDetectionBtn = document.getElementById('toggleDetectionBtn');
    let imgElement = toggleDetectionBtn.querySelector('img');
    toggleDetectionBtn.addEventListener('click', toggleDetection);

    async function toggleDetection() {
        if (!detectionRunning) {
            await startDetection();
            // El canvas se mostrará cuando haya algo que dibujar por drawLoop
            imgElement.src = "./imagenes/iconos/grabarB.svg";
            imgElement.className = "parpadeo";
        } else {
            stopDetection();
            canvas.style.display = 'none';
            imgElement.src = "./imagenes/iconos/grabarA.svg";
            imgElement.className = "";
        }
    }

    async function loadModels() {
        gif_carga_principal.style.display = 'block';
        try {
            if (!alprInstance) {
                // Los nombres de modelo son los que espera tu constructor de ALPR
                // y ALPR se encarga de las rutas a los .onnx
                alprInstance = new ALPR({
                    detectorModel: "yolo-v9-t-384-license-plates-end2end", // Puedes elegir el modelo aquí
                    ocrModel: "global_mobile_vit_v2_ocr"
                    // Si tu ALPR necesita una ruta base para los modelos, configúrala aquí
                    // ej: modelsBasePath: './models/' (o donde estén tus .onnx)
                });
                // Si ALPR tiene un método init() asíncrono, deberías llamarlo:
                // await alprInstance.init();
                console.log("ALPR instance creada y modelos listos.");
            }
        } catch (error) {
            alert("Error cargando modelos ALPR: " + error.message);
            console.error("Error cargando modelos ALPR:", error);
            gif_carga_principal.style.display = 'none';
            throw error; // Propagar el error para detener la inicialización si falla
        }
        gif_carga_principal.style.display = 'none';
    }

    let cameraTracks = [];
    let zoomLevel = 1;
    const zoomStep = 0.2;

    async function startDetection() {
        detectionRunning = true;
        video = document.getElementById('video');
        canvas = document.getElementById('canvas');
        ctx = canvas.getContext('2d');

        // Las dimensiones del canvas se ajustarán dinámicamente o al recibir la primera imagen procesada

        try {
            await loadModels(); // Carga la instancia ALPR

            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment" },
                audio: false
            });
            cameraTracks = cameraStream.getTracks();
            video.srcObject = cameraStream;

            video.onloadedmetadata = function () {
                const videoWidth = video.videoWidth;
                const videoHeight = video.videoHeight;

                container.style.width = `${videoWidth}px`;
                container.style.height = `${videoHeight}px`;
                video.style.width = "100%";
                video.style.height = "100%";

                // Ajustar el canvas al tamaño del video fuente una vez que los metadatos están cargados
                canvas.width = videoWidth;
                canvas.height = videoHeight;
                console.log(`Canvas and video initialized to: ${canvas.width}x${canvas.height}`);
            };

            video.oncanplay = () => { // Asegurarse que el video está listo para empezar a procesar frames
                if (!animationId) { // Evitar múltiples bucles si el evento se dispara varias veces
                    processFrame();
                    iniciarVerificacion();
                    iniciarDrawLopp(); // Inicia el bucle de dibujo
                    canvas.style.display = 'block'; // Mostrar el canvas ahora
                }
            };

        } catch (error) {
            alert("Error al iniciar la detección: " + error.message);
            console.error("Error al iniciar la detección:", error);
            stopDetection(); // Asegurarse de limpiar si falla el inicio
        }
    }

    function ajustarZoom(incremento) {
        if (!cameraTracks.length) {
            console.warn("No se ha encontrado una pista de video.");
            return;
        }
        let videoTrack = cameraTracks[0];
        let capabilities = videoTrack.getCapabilities();
        if (!capabilities.zoom) {
            return;
        }
        let newZoom = zoomLevel + incremento;
        newZoom = Math.max(capabilities.zoom.min, Math.min(capabilities.zoom.max, newZoom));
        videoTrack.applyConstraints({ advanced: [{ zoom: newZoom }] })
            .then(() => {
                zoomLevel = newZoom;
                // console.log("Zoom ajustado a:", zoomLevel);
            })
            .catch(error => console.error("Error al aplicar zoom:", error));
    }

    document.getElementById("zoomIn").addEventListener("click", () => ajustarZoom(zoomStep));
    document.getElementById("zoomOut").addEventListener("click", () => ajustarZoom(-zoomStep));

    function stopDetection() {
        detectionRunning = false;
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        detenerVerificacion();
        detenerDrawLopp();
        ctx.clearRect(0, 0, canvas.width, canvas.height); // Limpiar canvas al detener
        canvas.style.display = 'none'; // Ocultar canvas
        console.log("Detección detenida.");
    }

    function dataURLtoBlob(dataURL) {
        const parts = dataURL.split(';base64,');
        if (parts.length !== 2) {
            throw new Error('Formato de Data URL inválido');
        }
        const mimeType = parts[0].split(':')[1];
        const byteCharacters = atob(parts[1]);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        return new Blob([new Uint8Array(byteNumbers)], { type: mimeType });
    }

    async function verificarPlate(plateText, confianzaText, ubicacion, frameImageSrc) { // frameImage ahora es src
        try {
            const formData = new FormData();
            formData.append("plateText", plateText);
            formData.append("confianzaText", confianzaText);
            formData.append("ubicacion", ubicacion);

            // Si frameImageSrc es un Data URL, conviértelo a Blob
            if (frameImageSrc && typeof frameImageSrc === 'string' && frameImageSrc.startsWith('data:image')) {
                 const blob = dataURLtoBlob(frameImageSrc);
                 formData.append("frameImage", blob, "captura.png");
            } else if (frameImageSrc instanceof Blob) { // Si ya es un Blob (menos probable aquí)
                 formData.append("frameImage", frameImageSrc, "captura.png");
            } else {
                console.warn("No se proporcionó imagen válida para la verificación.");
            }


            const response = await fetch("./z/procesar_verificacion", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error("Error en la conexión con el servidor: " + response.statusText);
            }
            const textResponse = await response.text();
            try {
                return JSON.parse(textResponse);
            } catch (error) {
                console.error("No se pudo parsear JSON de la verificación:", textResponse);
                return { success: false, message: "Respuesta inválida del servidor." };
            }
        } catch (error) {
            console.error("Error en verificarPlate:", error);
            return { success: false, message: error.message };
        }
    }

    let divEstado_loop;
    async function verificarLoop() {
        for (let i = mispatentes.length - 1; i >= 0; i--) { // Iterar al revés por si se eliminan elementos
            let patente = mispatentes[i];
            if (!patente.verificado) {
                let ubicacionText = "";
                try {
                    const ubicacion = await obtenerUbicacion();
                    ubicacionText = `${ubicacion.latitud}, ${ubicacion.longitud}`;
                    patente.ubicacionText = ubicacionText; // Actualizar por si acaso
                } catch (error) {
                    console.warn("Error obteniendo la ubicación para verificación:", error.message);
                }

                // Pasar frameImage.src en lugar del elemento imagen completo
                const r = await verificarPlate(
                    patente.plateText,
                    patente.confianzaText,
                    ubicacionText,
                    patente.frameImageSrc // Usar la DataURL guardada
                );

                // Ya no necesitamos la imagen en el cliente después de enviarla (o si no se envió)
                patente.frameImageSrc = null; // Liberar memoria

                console.log("Respuesta del servidor (verificación):", r.message);

                divEstado_loop = document.getElementById(patente.plateText);
                if (divEstado_loop) {
                    patente.verificado = true;
                    const divCirculo = document.createElement("div");
                    if (r.buscado) { // Asumiendo que 'buscado' es el campo relevante
                        sonido_rojo();
                        divCirculo.className = "estado_rojo";
                    } else {
                        divCirculo.className = "estado_verde";
                    }
                    divEstado_loop.innerHTML = "";
                    divEstado_loop.appendChild(divCirculo);
                } else {
                    console.warn(`Elemento con ID ${patente.plateText} no encontrado en DOM para actualizar estado.`);
                }
            }
        }

        if (verificadorRunning) {
            setTimeout(verificarLoop, 5000);
        }
    }

    function iniciarVerificacion() {
        if (!verificadorRunning) {
            verificadorRunning = true;
            verificarLoop();
        }
    }

    function detenerVerificacion() {
        verificadorRunning = false;
    }

    async function obtenerUbicacion() {
        return new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                reject("Geolocalización no es compatible.");
                return;
            }
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    resolve({
                        latitud: position.coords.latitude,
                        longitud: position.coords.longitude
                    });
                },
                (error) => {
                    reject(`Error obteniendo ubicación: ${error.message}`);
                },
                { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
            );
        });
    }

    function limpiar_dibujo() {
        if (ctx && canvas) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }

    let ultimoDibujo = Date.now();
    let latestDetections = [];
    let latestFrameImage = null; // HTMLImageElement

    async function drawLoop() { // Marcada como async por drawDetections
        if (!drawLoopRunning) return;

        const ahora = Date.now();
        if (latestFrameImage && alprInstance) { // Asegurarse que alprInstance está listo
            ultimoDibujo = ahora;
            try {
                // Usar la función de dibujo de ALPR. Puede que necesite `latestDetections` como argumento
                // o que `predict` y `drawPredictions` estén más acoplados en tu biblioteca.
                // Asumo que `drawPredictions` toma la imagen y opcionalmente las detecciones precalculadas.
                // Si `drawPredictions` no toma `latestDetections`, las calculará internamente.
                const canvasConDetecciones = await alprInstance.drawPredictions(latestFrameImage, latestDetections);

                // Asegurar que el canvas principal tiene el tamaño correcto
                if (canvas.width !== canvasConDetecciones.width || canvas.height !== canvasConDetecciones.height) {
                    canvas.width = canvasConDetecciones.width;
                    canvas.height = canvasConDetecciones.height;
                }

                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(canvasConDetecciones, 0, 0);

            } catch (error) {
                console.error("Error en alprInstance.drawPredictions:", error);
                limpiar_dibujo(); // Limpiar si falla el dibujo
            }
            // Ya no reseteamos latestFrameImage o latestDetections aquí,
            // se actualizan en processFrame. El dibujo siempre usa lo más reciente.
        } else {
            if (ahora - ultimoDibujo > 330) { // Limpiar si no hay nada que dibujar por un tiempo
                limpiar_dibujo();
            }
        }
        requestAnimationFrame(drawLoop);
    }


    function iniciarDrawLopp() {
        if (!drawLoopRunning) {
            drawLoopRunning = true;
            drawLoop();
        }
    }

    function detenerDrawLopp() {
        drawLoopRunning = false;
    }

    // Nueva función para la lógica de zoom basada en detecciones de ALPR
    function adjustZoomFromDetections(detectionsFromALPR, capturedFrameWidth) {
        if (!detectionsFromALPR || detectionsFromALPR.length === 0 || !canvas) {
             // Opcional: Lógica para zoom out si no se detecta nada por un tiempo
            return;
        }

        let maxPlateWidthOnCanvas = 0;
        // Escala para convertir coordenadas de la imagen original al canvas actual (si son diferentes)
        // Asumimos que el canvas principal (donde se dibuja) tiene las dimensiones del video.
        const scaleX = canvas.width / capturedFrameWidth;

        detectionsFromALPR.forEach(item => {
            const { x1, x2 } = item.detection.boundingBox; // Coordenadas en la imagen original
            const plateWidthInOriginal = x2 - x1;
            const plateWidthOnCanvas = plateWidthInOriginal * scaleX; // Ancho de la placa como se vería en el canvas

            if (plateWidthOnCanvas > maxPlateWidthOnCanvas) {
                maxPlateWidthOnCanvas = plateWidthOnCanvas;
            }
        });

        // console.log(`Ancho máx de placa en canvas: ${maxPlateWidthOnCanvas.toFixed(0)}px`);

        if (maxPlateWidthOnCanvas > 0) { // Solo ajustar si se detectó alguna placa
            if (maxPlateWidthOnCanvas < 140) { // Si la placa más grande es muy pequeña
                // console.log(`Placa pequeña (${maxPlateWidthOnCanvas.toFixed(0)}px), aplicando zoom in.`);
                ajustarZoom(zoomStep);
            } else if (maxPlateWidthOnCanvas > 240) { // Si la placa más grande es muy grande
                // console.log(`Placa grande (${maxPlateWidthOnCanvas.toFixed(0)}px), aplicando zoom out.`);
                ajustarZoom(-zoomStep);
            }
        }
    }


    async function processFrame() {
        if (!detectionRunning || !alprInstance) { // Asegurar que alprInstance está lista
            if (detectionRunning) requestAnimationFrame(processFrame); // Intentar de nuevo si solo faltaba alprInstance
            return;
        }

        // Capturar el frame actual del video
        const frameImageElement = await captureFrame(); // Retorna un HTMLImageElement

        if (frameImageElement && frameImageElement.complete && frameImageElement.naturalWidth > 0) {
            try {
                // 1. Obtener predicciones de ALPR (detección + OCR)
                const alprResults = await alprInstance.predict(frameImageElement);

                // Guardar para el bucle de dibujo y lógica de zoom
                latestFrameImage = frameImageElement; // frameImageElement es el HTMLImageElement
                latestDetections = alprResults; // Resultados de ALPR

                // Llamar a la lógica de zoom aquí, después de obtener las detecciones
                // Necesitamos el ancho original del frameImageElement para el cálculo correcto
                adjustZoomFromDetections(alprResults, frameImageElement.naturalWidth);


                if (alprResults && alprResults.length > 0) {
                    for (const item of alprResults) {
                        const { detection, ocr } = item;
                        const plateText = ocr.text || "";
                        const detectionConfidence = detection.confidence || 0;
                        const ocrConfidence = ocr.confidence || 0;
                        // const boundingBox = detection.boundingBox; // {x1, y1, x2, y2}

                        let confianza_detec_patente = parseFloat((detectionConfidence * 100).toFixed(1));
                        let porcentajeOCR = parseFloat((ocrConfidence * 100).toFixed(1));
                        let confianzaText = `${confianza_detec_patente}%/${porcentajeOCR}%`;

                        if (
                            confianza_detec_patente > 70 &&
                            porcentajeOCR > 76 &&
                            !mispatentes.some(p => p.plateText === plateText) &&
                            plateText.length > 3 && // Un filtro básico de longitud
                            !plateText.includes('_') // Caracter de placeholder común en algunos OCRs
                        ) {
                            // Para agregarElemento, necesitamos la imagen como DataURL para el envío al backend
                            // y opcionalmente para mostrarla en la lista.
                            // Clonamos el frameImageElement para la lista (si es necesario mostrarlo ahí)
                            // y guardamos su src (DataURL) para la verificación.
                            const displayImage = frameImageElement.cloneNode(true);
                            displayImage.style.width = "100px"; // Ejemplo de tamaño para la lista
                            displayImage.style.height = "auto";

                            // El frameImageElement.src ya es una DataURL por cómo funciona captureFrame()
                            agregarElemento(plateText, confianzaText, "", displayImage.src, displayImage);
                        }
                    }
                }
            } catch (error) {
                console.error("Error en processFrame con ALPR:", error);
            }
        } else if (frameImageElement) {
             console.warn("Frame capturado pero no está completo o no tiene dimensiones.");
        }


        // Programar el siguiente frame
        if (detectionRunning) {
            animationId = requestAnimationFrame(processFrame);
        }
    }

    function actualizar_contador() {
        const divContador = document.getElementById("divContador");
        divContador.textContent = mispatentes.length;
    }

    // Modificado para aceptar frameImageSrc (DataURL) y frameImageElement (para mostrar en DOM si se desea)
    function agregarElemento(plateText, confianzaText, ubicacionText, frameImageSrc, frameImageElement = null) {
        const divImagen = document.createElement("div");
        divImagen.classList.add("c_img_caputura");
        if (frameImageElement) { // Si se pasa un elemento imagen para mostrar
            divImagen.appendChild(frameImageElement);
        } else { // Fallback si solo se pasa el src, crear un img nuevo
            const img = new Image();
            img.src = frameImageSrc;
            img.style.width = "100px";
            img.style.height = "auto";
            divImagen.appendChild(img);
        }

        const divPatente = document.createElement("div");
        divPatente.textContent = plateText;

        const divConfianzas = document.createElement("div");
        divConfianzas.textContent = confianzaText;

        const divEstado = document.createElement("div");
        divEstado.id = plateText;
        const divCirculo = document.createElement("div");
        divCirculo.classList.add("estado_naranjo");
        divEstado.appendChild(divCirculo);

        const nuevoElemento = document.createElement("div");
        nuevoElemento.classList.add("grid-contenedor");
        nuevoElemento.appendChild(divImagen);
        nuevoElemento.appendChild(divPatente);
        nuevoElemento.appendChild(divConfianzas);
        nuevoElemento.appendChild(divEstado);

        const resultadoDiv = document.getElementById("resultado");
        resultadoDiv.insertBefore(nuevoElemento, resultadoDiv.firstChild); // Añadir al principio


        mispatentes.push({
            plateText: plateText,
            confianzaText: confianzaText,
            ubicacionText: ubicacionText,
            frameImageSrc: frameImageSrc, // Guardar DataURL para verificación
            verificado: false
        });

        sonido_gris();
        actualizar_contador();
    }

    async function captureFrame() {
        if (!video || !video.videoWidth || !video.videoHeight || video.readyState < video.HAVE_wystarczajaco_DANYCH) { // HAVE_ENOUGH_DATA
            // console.warn("Video no listo para capturar frame.");
            return null;
        }

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true }); // willReadFrequently para optimizar toDataURL
        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => {
                console.error("Error al cargar imagen desde DataURL de canvas.");
                resolve(null); // Resolver con null si hay error
            }
            img.src = tempCanvas.toDataURL("image/png"); // ALPR probablemente espera un HTMLImageElement
        });
    }

    // Inicialización de la cámara (opcional, si no se hace al hacer clic en "grabar")
    // Podrías llamar a `loadModels()` aquí para pre-cargar al inicio de la página.
    // loadModels().catch(err => console.error("Fallo en precarga de modelos:", err));
});