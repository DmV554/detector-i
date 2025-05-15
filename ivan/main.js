
import { loadDetectorModel, detectPlates, cropImage } from './detector.js';
import { loadOcrModel, runOCRInference } from './ocr.js';


ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
const canUseThreads = self.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined';

if (canUseThreads) {
  // limitar a 4 hilos para no saturar móviles de gama baja
  ort.env.wasm.numThreads = Math.min(4, Math.max(1, navigator.hardwareConcurrency - 1));
}
ort.env.wasm.simd       = true;


window.addEventListener("load", function () {
    let video, canvas, ctx;
    let detectorSession, ocrSession;
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
    const  superCccCam = document.getElementById('super-ccc-cam');
    const  toggleExpandirBtn = document.getElementById('toggleExpandirBtn');
    let imgExpandir = toggleExpandirBtn.querySelector('img');
    toggleExpandirBtn.addEventListener('click', toggleExpandir);
    function toggleExpandir() {
        if (!expandido) {
            imgExpandir.src = "./imagenes/iconos/expandirB.svg"; // Cambia la ruta de la imagen
            superCccCam.className = "super-ccc-cam_expan padding_A";
            expandido = true;
        }else {
            imgExpandir.src = "./imagenes/iconos/expandirA.svg";
            superCccCam.className = "super-ccc-cam base_float";
            expandido = false;
        }
    }

    let VideoOculto = false;
    let  ccc_cam = document.getElementById('ccc_cam');
    const  toggleCamBtn = document.getElementById('toggleCamBtn');
    let imgVideoOculto = toggleCamBtn.querySelector('img');
    toggleCamBtn.addEventListener('click', toggleOcultar);
    function toggleOcultar() {
        if (!VideoOculto) {
            imgVideoOculto.src = "./imagenes/iconos/ocultarCamB.svg"; // Cambia la ruta de la imagen
            ccc_cam.style.display = "none";
            VideoOculto = true;
        }else {
            imgVideoOculto.src = "./imagenes/iconos/ocultarCamA.svg";
            ccc_cam.style.display = "block";
            VideoOculto = false;
        }
    }


    const  toggleDetectionBtn = document.getElementById('toggleDetectionBtn');
    let imgElement = toggleDetectionBtn.querySelector('img');
    toggleDetectionBtn.addEventListener('click', toggleDetection);

    async function toggleDetection() {
        if (!detectionRunning) {
            await startDetection();
            canvas.style.display = 'block';
            imgElement.src = "./imagenes/iconos/grabarB.svg"; // Cambia la ruta de la imagen
            imgElement.className = "parpadeo";
        }else {
            stopDetection();
            canvas.style.display = 'none';
            imgElement.src = "./imagenes/iconos/grabarA.svg";
            imgElement.className = "";
            
        }
    }


    async function loadModels() {
        gif_carga_principal.style.display = 'block'; 

        try {
            if (!detectorSession) {
                detectorSession = await loadDetectorModel('./z/yolo-onnx/yolo-v9-t-384-license-plates-end2end.onnx');
            }
        } catch (error) {
            alert("Error cargando el modelo del detector:")
            console.error("Error cargando el modelo del detector:", error);
        }

        try {
            if (!ocrSession) {
                ocrSession = await loadOcrModel('./z/ocr-onnx/global_mobile_vit_v2_ocr.onnx');
            }
        } catch (error) {
            alert("Error cargando el modelo:")
            console.error("Error cargando el modelo:", error);
        }

        gif_carga_principal.style.display = 'none';
    }


    let cameraTracks = [];
    let zoomLevel = 1; // Nivel inicial de zoom
    const zoomStep = 0.2; // Incremento/decremento del zoom
    async function startDetection() {

        detectionRunning = true;
        video = document.getElementById('video');
        canvas = document.getElementById('canvas');
        ctx = canvas.getContext('2d');

        canvas.width = container.videoWidth;
        canvas.height = container.videoHeight;

        await loadModels();
        
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment" },
                audio: false
            });
            cameraTracks = cameraStream.getTracks();
            video.srcObject = cameraStream;
            video.onloadedmetadata = function () {
                // Ajustar dimensiones del contenedor al tamaño del video
                container.style.width = `${video.videoWidth}px`;
                container.style.height = `${video.videoHeight}px`;

                // Ajustar el tamaño del video y el canvas
                video.style.width = "100%";
                video.style.height = "100%";
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            };

            processFrame();

            //verificarLoop();
            iniciarVerificacion();


            iniciarDrawLopp();

        } catch (error) {
            alert("Error al acceder a la cámara:")
            console.error("Error al acceder a la cámara:", error);
        }

        
    }

    function ajustarZoom(incremento) {
        if (!cameraTracks.length) {
            console.warn("No se ha encontrado una pista de video.");
            return;
        }
    
        let videoTrack = cameraTracks[0]; // Tomamos la primera pista de video
        let capabilities = videoTrack.getCapabilities(); // Obtener capacidades de la cámara
    
        if (!capabilities.zoom) {
            //console.warn("La cámara no soporta zoom.");
            return;
        }
    
        let newZoom = zoomLevel + incremento;
        newZoom = Math.max(capabilities.zoom.min, Math.min(capabilities.zoom.max, newZoom));
    
        videoTrack.applyConstraints({ advanced: [{ zoom: newZoom }] });
        zoomLevel = newZoom;
    }
    
    // Agregar eventos a los botones de zoom
    document.getElementById("zoomIn").addEventListener("click", () => ajustarZoom(zoomStep));
    document.getElementById("zoomOut").addEventListener("click", () => ajustarZoom(-zoomStep));

    function pausarGrabacion() {
        if (cameraStream) {
            cameraTracks.forEach(track => track.enabled = false); // Deshabilita los tracks sin detenerlos
            console.log("Grabación pausada");
        }
    }

    function reanudarGrabacion() {
        if (cameraStream) {
            cameraTracks.forEach(track => track.enabled = false); // Reactiva los tracks
            console.log("Grabación reanudada");
        }
    }


    function stopDetection() {
        detectionRunning = false;
        if (cameraStream) {
            const tracks = cameraStream.getTracks();
            tracks.forEach(track => track.stop());
            cameraStream = null;
        }
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }

        detenerVerificacion();
        detenerDrawLopp();
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

    async function verificarPlate(plateText,confianzaText, ubicacion, frameImage) {
        try {
            const formData = new FormData();
            formData.append("plateText", plateText);
            formData.append("confianzaText", confianzaText);
            formData.append("ubicacion", ubicacion);

            const blob = dataURLtoBlob(frameImage.src);
            formData.append("frameImage", blob, "captura.png");

            const response = await fetch("./z/procesar_verificacion", {
                method: "POST",
                body: formData
            });

            
            //console.log(response);
    
            if (!response.ok) {
                throw new Error("Error en la conexión con el servidor");
            }
    
            const textResponse = await response.text();
            //console.log("Respuesta del servidor:" + JSON.parse(textResponse)); // Se espera que el servidor devuelva { "success": true } o { "success": false }

            try {
                return JSON.parse(textResponse);
            } catch (error) {
                console.error("No se pudo parsear JSON:", textResponse);
                return false;
            }
        } catch (error) {
            console.error("Error:", error);
            return false;
        }
    }





    let divEstado_loop;
    async function verificarLoop() {
        for (let patente of mispatentes) {
            if (!patente.verificado) { // Solo verificar las que no han sido verificadas


                /* */
                let ubicacionText ="";

                try {
                    let ubicacion = await obtenerUbicacion();
                    ubicacionText = `${ubicacion.latitud}, ${ubicacion.longitud}`;
                } catch (error) {
                    console.error("Error obteniendo la ubicación:", error);
                }/* */


                const r = await verificarPlate(
                    patente.plateText,
                    patente.confianzaText,
                    ubicacionText,
                    patente.frameImage
                );

                patente.frameImage = null;


                console.log("Respuesta del servidor:" + r.message);

                

                divEstado_loop = document.getElementById(patente.plateText);
                if (divEstado_loop) {
                    patente.verificado = true;
                    
                    if (r.buscado) {
                        sonido_rojo();

                        const divCirculo = document.createElement("div");
                        divCirculo.classList.add("estado_rojo");
                        
                        divEstado_loop.innerHTML = "";
                        divEstado_loop.appendChild(divCirculo);
                    } else {
                        
                        const divCirculo = document.createElement("div");
                        divCirculo.classList.add("estado_verde");

                        divEstado_loop.innerHTML = "";
                        divEstado_loop.appendChild(divCirculo);
                    }
                } else {
                    console.warn(`Elemento con ID ${patente.plateText} no encontrado en mispatentes`);
                }
    
            }
        }
    
        if (verificadorRunning){
            setTimeout(verificarLoop, 5000);
        } 
        
    }
    
    function iniciarVerificacion() {
        if (!verificadorRunning) {
            verificadorRunning = true;
            verificarLoop(); // Comenzar la ejecución
        }
    }

    function detenerVerificacion() {
        verificadorRunning = false;
    }


    async function obtenerUbicacion() {
        return new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                reject("Geolocalización no es compatible con este navegador.");
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
                }
            );
        });
    }

    /*
    pausarGrabacion();
    reanudarGrabacion();
    */


    function limpiar_dibujo() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    let ultimoDibujo = Date.now();
    
    let latestDetections = []; 
    let latestFrameImage = null; // Guardamos el último frame procesado

    // Loop independiente para dibujar en tiempo real
    function drawLoop() {

        const ahora = Date.now();
        if (latestFrameImage && latestDetections.length > 0) {
            ultimoDibujo = ahora;
            drawDetections(latestDetections, latestFrameImage);

            latestDetections = []; 
            latestFrameImage = null;

        }else {
            if (ahora - ultimoDibujo > 330) {
                limpiar_dibujo();
            }
        } 

        if (drawLoopRunning){
            requestAnimationFrame(drawLoop);
        } 
    }


    function iniciarDrawLopp() {
        if (!drawLoopRunning) {
            drawLoopRunning = true;
            drawLoop(); // Comenzar la ejecución
        }
    }

    function detenerDrawLopp() {
        drawLoopRunning = false;
    }

    async function processFrame() {
        if (!detectionRunning || animationId) return;

        const frameImage = await captureFrame();
        
        if (frameImage) {
            const detectedObjects = await detectPlates(frameImage, detectorSession);
            
            

            if (detectedObjects.length > 0) {
                // Recortar imágenes de las detecciones
                const croppedImages = await Promise.all(cropImage(frameImage, detectedObjects));

                // Filtrar imágenes válidas
                const validImages = croppedImages.map((cimg, index) => cimg ? { cimg, index } : null).filter(Boolean);

                if (validImages.length > 0) {
                    // Ejecutar OCR en paralelo para imágenes válidas
                    const ocrResults = await Promise.all(
                        validImages.map(({ cimg, index }) =>
                            runOCRInference(cimg, ocrSession, 6).then(result => ({ ...result, index }))
                        )
                    );

                    // Procesar resultados de OCR
                    for (const { plateText, confidence, index } of ocrResults) {
                        const detectionConfidence = detectedObjects[index]?.confidence || 0;

                        let confianza_detec_patente = parseFloat((detectionConfidence * 100).toFixed(1));
                        let porcentajeOCR = parseFloat((confidence * 100).toFixed(1));
                        let confianzaText = `${confianza_detec_patente}%/${porcentajeOCR}%`;

                        if (detectedObjects[index]) {
                            detectedObjects[index].plateText = plateText;
                            detectedObjects[index].confianzaText = confianzaText;
                        }

                        if (
                            confianza_detec_patente > 70 &&
                            porcentajeOCR > 76 &&
                            !mispatentes.some(patente => patente.plateText === plateText) &&
                            !plateText.includes('_')
                        ) {
                           

                            
                            let ubicacionText = "";

                            agregarElemento(plateText, confianzaText, ubicacionText, frameImage);
                        }
                    }
                }
            }

            // Guardamos el frame más reciente junto con las detecciones
            latestFrameImage = frameImage;
            latestDetections = detectedObjects; 

            
        }

        // Programar el siguiente frame solo si `detectionRunning` sigue activo
        if (detectionRunning) {
            animationId = requestAnimationFrame(() => {
                animationId = null;
                processFrame();
            });
        }
    }


    
    function actualizar_contador() {
        const divContador = document.getElementById("divContador");
        divContador.textContent = mispatentes.length;
    }


    function agregarElemento(plateText, confianzaText, ubicacionText, frameImage) {

        const divImagen = document.createElement("div");
        divImagen.classList.add("c_img_caputura");
        if (frameImage instanceof HTMLImageElement && frameImage.complete) {
            divImagen.appendChild(frameImage);
        } else {
            console.error("frameImage no es una imagen válida o no ha cargado.");
        }


        const divPatente = document.createElement("div");
        divPatente.textContent = plateText;
    
        const divConfianzas = document.createElement("div");
        divConfianzas.textContent = confianzaText;

    
        
        const divEstado = document.createElement("div");
        divEstado.id = plateText;//id estado
        const divCirculo = document.createElement("div");
        divCirculo.classList.add("estado_naranjo");
        divEstado.appendChild(divCirculo);


        // Crear el contenedor del nuevo elemento con grid
        const nuevoElemento = document.createElement("div");
        nuevoElemento.classList.add("grid-contenedor");
    
        // Agregar los divs al contenedor principal
        nuevoElemento.appendChild(divImagen);
        nuevoElemento.appendChild(divPatente);
        nuevoElemento.appendChild(divConfianzas);
        nuevoElemento.appendChild(divEstado);
    
        // Agregar el nuevo elemento al contenedor padre
        document.getElementById("resultado").appendChild(nuevoElemento);
    
        mispatentes.push({
            plateText: plateText,
            confianzaText: confianzaText,
            ubicacionText: ubicacionText,
            frameImage: frameImage, 
            verificado: false // Inicialmente no verificado
        });

        sonido_gris();
        actualizar_contador();
    }


    async function captureFrame() {
        if (!video.videoWidth || !video.videoHeight) return null;

        
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve(img); // Espera a que la imagen cargue
            img.src = tempCanvas.toDataURL("image/png");
        });
    }

    /**
     * Dibuja las detecciones (bounding boxes) en el canvas principal.
     */


    let zoomAutoIntentos = 0;
    const maxZoomIntentos = 5;

    function drawDetections(detections, frameImage) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const scaleX = canvas.width / frameImage.width;
        const scaleY = canvas.height / frameImage.height;

        let ancho_final=0;

        detections.forEach((d) => {
            const { x1, y1, x2, y2 } = d.boundingBox;
            const sx1 = x1 * scaleX;
            const sy1 = y1 * scaleY;
            const sx2 = x2 * scaleX;
            const sy2 = y2 * scaleY;

            const ancho = sx2 - sx1;
            //console.log(`Ancho de la patente: ${ancho.toFixed(2)} píxeles`);

            const posicionTexto = 6; // Ajusta la altura del texto
            const posicionFondo = 4; // Ajusta la altura del fondo
            const padding = 4; // Espaciado entre el texto y el borde del fondo

            ctx.beginPath();
            ctx.lineWidth = 3;
            ctx.strokeStyle = 'red';
            ctx.rect(sx1, sy1, sx2 - sx1, sy2 - sy1);
            ctx.stroke();

            ctx.font = '14px Arial';
            const text = `${d.plateText || ""}   ${d.confianzaText || ""}`;
            const textWidth = ctx.measureText(text).width;
            const textHeight = 16; // Aproximado para 14px de fuente

            // Dibujar fondo rojo con padding y posición independiente
            ctx.fillStyle = 'red';
            ctx.fillRect(sx1 - padding + 2, sy1 - posicionFondo - textHeight - padding, textWidth + padding * 2, textHeight + padding * 2);

            // Dibujar el texto en blanco con posición independiente
            ctx.fillStyle = 'white';
            ctx.fillText(text, sx1 + 3, sy1 - posicionTexto);


            ctx.fillStyle = 'yellow';
            ctx.font = '12px Arial';
            ctx.fillText(`${ancho.toFixed(0)}px`, sx1 + 5, sy2 + 14);

            let redondeado = Number(ancho.toFixed(0));

            if(ancho_final<redondeado){
                ancho_final=redondeado;
            }

        });

        //console.log(`patente mas grande: ${ancho_final.toFixed(0)} píxeles`);
        /*
        if (ancho_final < targetWidth && zoomAutoIntentos < maxZoomIntentos) {
            console.log(`Ancho de patente muy pequeño (${ancho_final}px), aplicando zoom automático.`);
            ajustarZoom(zoomStep);
            zoomAutoIntentos++;
        } else {
            zoomAutoIntentos = 0; // reiniciar si ya estamos bien
        }
        */

        if (ancho_final < 140) {
            //console.log(`Ancho de patente muy pequeño (${ancho_final}px), aplicando zoom automático.`);
            ajustarZoom(zoomStep);
        } else if(ancho_final>240) {
            ajustarZoom(-zoomStep);
        }
    }

});