/**
 * Base module - Define las clases e interfaces básicas.
 */

import { LicensePlateDetector } from "../../detection/openimageclaude.js"
import { OnnxOcrRecognizer } from "../../ocr/inference/onnx-ocr-inference.js";

/**
 * Representa un cuadro delimitador para una detección.
 */
class BoundingBox {
  /**
   * @param {number} x1 - Coordenada x superior izquierda
   * @param {number} y1 - Coordenada y superior izquierda
   * @param {number} x2 - Coordenada x inferior derecha
   * @param {number} y2 - Coordenada y inferior derecha
   */
  constructor(x1, y1, x2, y2) {
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
  }
}

/**
 * Resultado de una detección de placa.
 */
class DetectionResult {
  /**
   * @param {string} label - Etiqueta de la detección
   * @param {number} confidence - Confianza de la detección
   * @param {BoundingBox} boundingBox - Cuadro delimitador
   */
  constructor(label, confidence, boundingBox) {
    this.label = label;
    this.confidence = confidence;
    this.boundingBox = boundingBox;
  }
}

/**
 * Resultado del reconocimiento OCR.
 */
class OcrResult {
  /**
   * @param {string} text - Texto reconocido
   * @param {number|Array<number>} confidence - Confianza del reconocimiento
   */
  constructor(text, confidence) {
    this.text = text;
    this.confidence = confidence;
  }
}

/**
 * Clase base para detectores de placas.
 */
class BaseDetector {
  /**
   * Realiza la detección en un fotograma.
   * @param {cv.Mat} frame - Imagen de entrada
   * @returns {Array<DetectionResult>} Lista de detecciones
   */
  predict(frame) {
    throw new Error("El método predict debe ser implementado por las subclases");
  }
}

/**
 * Clase base para OCR.
 */
class BaseOCR {
  /**
   * Realiza OCR en una imagen recortada de placa.
   * @param {cv.Mat} croppedPlate - Imagen recortada de la placa
   * @returns {OcrResult|null} Resultado del OCR o null si no se pudo procesar
   */
  predict(croppedPlate) {
    throw new Error("El método predict debe ser implementado por las subclases");
  }
}

/**
 * DefaultDetector - Detector predeterminado utilizando modelos ONNX.
 */
class DefaultDetector extends BaseDetector {
  /**
   * @param {string} modelName - Nombre del modelo detector
   * @param {number} confThresh - Umbral de confianza
   * @param {Array} providers - Proveedores de ejecución ONNX
   * @param {Object} sessOptions - Opciones de sesión ONNX
   * @param detectorModelPath
   */
  constructor(modelName = "yolo-v9-t-384-license-plate-end2end", confThresh = 0.4, providers = null, sessOptions = null, detectorModelPath = "") {
    super();

    console.log(detectorModelPath);
    this.detector = new LicensePlateDetector({
        modelsPath: detectorModelPath,
      detectionModel: modelName,
      confThresh: confThresh,
      providers: providers,
    });
  }

  /**
   * @param {cv.Mat} frame - Imagen de entrada
   * @returns {Array<DetectionResult>} Lista de detecciones
   */
  async predict(frame) {
    const detections = await this.detector.predict(frame);
    console.log(detections)
    return detections.map(detection => {
      return new DetectionResult(
        detection.label,
        detection.confidence,
        new BoundingBox(
          detection.boundingBox.x1,
          detection.boundingBox.y1,
          detection.boundingBox.x2,
          detection.boundingBox.y2
        )
      );
    });
  }
}

/**
 * DefaultOCR - OCR predeterminado utilizando modelos ONNX.
 */
class DefaultOCR extends BaseOCR {
  /**
   * @param {string} hubOcrModel - Nombre del modelo OCR
   * @param {string} device - Dispositivo a utilizar ("cuda", "cpu", "auto")
   * @param {Array} providers - Proveedores de ejecución ONNX
   * @param {Object} sessOptions - Opciones de sesión ONNX
   * @param {string} modelPath - Ruta a un archivo de modelo OCR personalizado
   * @param {string} configPath - Ruta a un archivo de configuración personalizado
   * @param {boolean} forceDownload - Si es true, fuerza la descarga del modelo
   */
  constructor(
    hubOcrModel = "global-plates-mobile-vit-v2-model",
    device = "auto",
    providers = null,
    sessOptions = null,
    modelPath = null,
    configPath = null,
    forceDownload = false
  ) {
    super();
    // Asumimos que ONNXPlateRecognizer está disponible como una implementación JS externa
    this.ocrModel = new OnnxOcrRecognizer({
      hubOcrModel: hubOcrModel,
      device: device,
      providers: providers,
      sessOptions: sessOptions,
      modelPath: modelPath,
      configPath: configPath,
      forceDownload: forceDownload
    });
  }

  /**
   * @param {cv.Mat} croppedPlate - Imagen recortada de la placa
   * @returns {OcrResult|null} Resultado del OCR o null si no se pudo procesar
   */
  async predict(croppedPlate) {
    if (!croppedPlate || croppedPlate.empty()) {
      return null;
    }


    // Convertir a escala de grises usando OpenCV.js
    const grayPlate = new cv.Mat();
    cv.cvtColor(croppedPlate, grayPlate, cv.COLOR_BGR2GRAY);

    // Ejecutar OCR
      await this.ocrModel.initialize("global-plates-mobile-vit-v2-model")
      const res = await this.ocrModel.run(grayPlate, true);
    //const [plateText, probabilities] = this.ocrModel.run(grayPlate, true);

    // Limpiar
    //grayPlate.delete();

    let plateTextArray, confidenceObj;
    if (Array.isArray(res) && res.length === 2) {
      [plateTextArray, confidenceObj] = res;
    } else {
      // Si no es array, adapta según la forma real de `res`
      plateTextArray  = res.textArray    || [];
      confidenceObj   = res.probabilities || res;
    }

    // 5) Extraer texto final y media de confianza
    const rawText = Array.isArray(plateTextArray)
      ? plateTextArray.pop()
      : String(plateTextArray);
    const text = rawText.replace(/_/g, "");
    const meanConf = typeof confidenceObj.mean === 'function'
      ? confidenceObj.mean()
      : Number(confidenceObj);

    return new OcrResult(text, meanConf);
  }
}

/**
 * Resultado de ALPR que contiene detección y OCR.
 */
class ALPRResult {
  /**
   * @param {DetectionResult} detection - Resultado de la detección
   * @param {OcrResult|null} ocr - Resultado del OCR
   */
  constructor(detection, ocr) {
    this.detection = detection;
    this.ocr = ocr;
  }
}

/**
 * ALPR - Sistema de Reconocimiento Automático de Matrículas.
 */
export default class ALPR {
  /**
   * @param {BaseDetector} detector - Instancia de BaseDetector
   * @param {BaseOCR} ocr - Instancia de BaseOCR
   * @param {string} detectorModel - Nombre del modelo detector
   * @param {number} detectorConfThresh - Umbral de confianza para el detector
   * @param {Array} detectorProviders - Proveedores de ejecución para el detector
   * @param {Object} detectorSessOptions - Opciones de sesión para el detector
   * @param {string} ocrModel - Nombre del modelo OCR
   * @param {string} ocrDevice - Dispositivo para ejecutar el modelo OCR
   * @param {Array} ocrProviders - Proveedores de ejecución para OCR
   * @param {Object} ocrSessOptions - Opciones de sesión para OCR
   * @param {string} ocrModelPath - Ruta personalizada para el modelo OCR
   * @param {string} ocrConfigPath - Ruta personalizada para la configuración OCR
   * @param {boolean} ocrForceDownload - Forzar descarga del modelo OCR
   * @param detectorModelPath
   */
  constructor({
    detector = null,
    ocr = null,
    detectorModel = "yolo-v9-t-384-license-plate-end2end",
    detectorConfThresh = 0.4,
    detectorProviders = null,
    detectorSessOptions = null,
    ocrModel = "global-plates-mobile-vit-v2-model",
    ocrDevice = "auto",
    ocrProviders = null,
    ocrSessOptions = null,
    ocrModelPath = null,
    ocrConfigPath = null,
    ocrForceDownload = false,
      detectorModelPath = "./models"
  } = {}) {
    // Inicializar el detector
    this.detector = detector || new DefaultDetector(
      detectorModel,
      detectorConfThresh,
      detectorProviders,
      detectorSessOptions,
        detectorModelPath = "./models"
    );

    // Inicializar el OCR
    this.ocr = ocr || new DefaultOCR(
      ocrModel,
      ocrDevice,
      ocrProviders,
      ocrSessOptions,
      ocrModelPath,
      ocrConfigPath,
      ocrForceDownload
    );
  }

  /**
   * Carga una imagen desde un archivo o URL.
   * @param {string|Blob|File} source - Fuente de la imagen (URL, Blob o File)
   * @returns {Promise<cv.Mat>} Imagen cargada como cv.Mat
   */
  /**
   * Convierte un HTMLImageElement en cv.Mat usando un canvas intermedio.
   * @param {HTMLImageElement} imgEl
   * @returns {cv.Mat}
   */
  _imgElementToMat(imgEl) {
    const canvas = document.createElement('canvas');
    canvas.width = imgEl.naturalWidth;
    canvas.height = imgEl.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgEl, 0, 0);
    const mat = cv.imread(canvas);
    canvas.remove();
    return mat;
  }

  /**
   * Carga una imagen desde diferentes fuentes y la devuelve como cv.Mat.
   * @param {string|Blob|File|HTMLImageElement|cv.Mat} source
   * @returns {Promise<cv.Mat>}
   */
  async loadImage(source) {

    const isInWorker = typeof self.document === 'undefined';

    if (source instanceof (isInWorker ? OffscreenCanvas : HTMLCanvasElement)) { // Adaptar al entorno
        return source;
    }
    if (source instanceof cv.Mat) { // Si ya es un Mat (desde el worker)
        // El detector espera un Canvas/Image. OCR espera Mat.
        // Esto es ineficiente. Idealmente, el detector también tomaría Mat.
        // Para ahora, si estamos en worker, creamos OffscreenCanvas.
        if (isInWorker) {
            const offscreenCanvas = new OffscreenCanvas(source.cols, source.rows);
            cv.imshow(offscreenCanvas, source); // cv.imshow puede funcionar con OffscreenCanvas si OpenCV está compilado para ello
            return offscreenCanvas;
        } else { // Hilo principal
            const canvas = document.createElement('canvas');
            canvas.width = source.cols;
            canvas.height = source.rows;
            cv.imshow(canvas, source);
            return canvas;
        }
    }
    if (source instanceof ImageData && isInWorker) {
        const offscreenCanvas = new OffscreenCanvas(source.width, source.height);
        offscreenCanvas.getContext('2d').putImageData(source, 0, 0);
        return offscreenCanvas;
    }

    // Si ya viene un canvas, lo devolvemos directamente
    if (source instanceof HTMLCanvasElement) {
      return source;
    }

    // Si viene un cv.Mat, volcamos al canvas
    if (source instanceof cv.Mat) {
      const cvCanvas = document.createElement('canvas');
      cv.imshow(cvCanvas, source);
      return cvCanvas;
    }

    // Función auxiliar: dibuja un <img> en un <canvas>
    const imgToCanvas = (img) => {
      const c = document.createElement('canvas');
      c.width  = img.naturalWidth || img.width;
      c.height = img.naturalHeight || img.height;
      c.getContext('2d').drawImage(img, 0, 0);
      return c;
    };

    // Crea un <img> desde URL o DataURL
    const loadImgEl = (url) => new Promise((res, rej) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload  = () => res(img);
      img.onerror = () => rej(new Error(`No se pudo cargar imagen: ${url}`));
      img.src     = url;
    });

     let imgEl;
    if (typeof source === 'string') {
      imgEl = await loadImgEl(source);
    } else if (source instanceof Blob || source instanceof File) {
      const dataURL = await new Promise((res, rej) => { /* ... */ });
      imgEl = await loadImgEl(dataURL);
    } else if (source instanceof HTMLImageElement) {
      imgEl = source;
    } else {
      throw new TypeError('Formato de imagen no soportado en loadImage (contexto principal)');
    }
    return imgToCanvas(imgEl); // imgToCanvas debe usar OffscreenCanvas si está en worker
  }


  /**
   * Detecta y reconoce matrículas en una imagen.
   * @param {string|Blob|File|HTMLImageElement|HTMLCanvasElement|cv.Mat} frame
   * @returns {Promise<ALPRResult[]>}
   */

 async predict(frameInput) {
    let inputForDetector;
    let matForOcr;
    let mustDeleteMatForOcr = false;
    let mustDeleteTempMat = false; // Para el mat RGBA temporal si se crea

    const isInWorker = typeof self.document === 'undefined';

    if (frameInput instanceof cv.Mat) {
        matForOcr = frameInput; // OCR usa el Mat directamente

        if (isInWorker) {
            // Convertir cv.Mat (frameInput) a ImageData para el detector
            // YoloV9ObjectDetector (en openimageclaude.js) puede manejar ImageData a través de Utils.letterbox.
            let tempMatRgba = new cv.Mat(); // Mat temporal para la conversión a RGBA
            mustDeleteTempMat = true;

            if (frameInput.channels() === 1) {
                cv.cvtColor(frameInput, tempMatRgba, cv.COLOR_GRAY2RGBA);
            } else if (frameInput.channels() === 3) {
                // Asumiendo que el frameInput (cv.Mat) que viene del worker es BGR
                // (cv.matFromImageData usualmente produce RGBA, pero si se manipula podría ser BGR)
                // ImageData espera datos en formato RGBA.
                cv.cvtColor(frameInput, tempMatRgba, cv.COLOR_BGR2RGBA);
            } else if (frameInput.channels() === 4) {
                // Si ya es RGBA (CV_8UC4), podemos usarlo directamente.
                // No necesitamos copiar a tempMatRgba a menos que queramos asegurar un clon.
                // Por simplicidad, vamos a asumir que la conversión es necesaria o que tempMatRgba se usará.
                // Si frameInput ya es el formato correcto y no se modifica, podríamos optimizar.
                frameInput.copyTo(tempMatRgba);
            } else {
                console.error("ALPR.predict (worker): Número de canales no soportado en frameInput (cv.Mat) para convertir a ImageData:", frameInput.channels());
                if (!tempMatRgba.isDeleted()) tempMatRgba.delete();
                return []; // Retornar vacío si no se puede procesar
            }

            // Crear ImageData a partir del Mat RGBA.
            // Asegurarse de que tempMatRgba sea del tipo CV_8UC4 (unsigned 8-bit, 4 channels).
            // Las funciones cvtColor a RGBA suelen producir esto.
            if (tempMatRgba.type() !== cv.CV_8UC4) {
                 console.warn("ALPR.predict (worker): tempMatRgba no es CV_8UC4, es tipo:", tempMatRgba.type(), ". Intentando conversión forzada.");
                 // Podríamos intentar convertirlo si no es el tipo correcto, aunque cvtColor debería haberlo hecho.
                 // Esto es un fallback, idealmente cvtColor ya lo deja en CV_8UC4.
                 let finalRgbaMat = new cv.Mat();
                 tempMatRgba.convertTo(finalRgbaMat, cv.CV_8UC4);
                 if (!tempMatRgba.isDeleted()) tempMatRgba.delete(); // Borrar el intermedio
                 tempMatRgba = finalRgbaMat; // Reasignar
            }

            // Si tempMatRgba está vacío después de las conversiones, hay un problema.
            if (tempMatRgba.empty()) {
                console.error("ALPR.predict (worker): tempMatRgba está vacío después de intentar la conversión de color.");
                if (!tempMatRgba.isDeleted()) tempMatRgba.delete();
                return [];
            }

            // Crear el objeto ImageData
            try {
                inputForDetector = new ImageData(new Uint8ClampedArray(tempMatRgba.data), tempMatRgba.cols, tempMatRgba.rows);
            } catch (imgDataError) {
                console.error("ALPR.predict (worker): Error creando ImageData desde tempMatRgba.", imgDataError);
                console.error("Detalles de tempMatRgba: cols=", tempMatRgba.cols, "rows=", tempMatRgba.rows, "type=", tempMatRgba.type(), "data length=", tempMatRgba.data.length);
                if (!tempMatRgba.isDeleted()) tempMatRgba.delete();
                return [];
            }
            // tempMatRgba se borrará en el bloque finally general de este método predict.

        } else { // Hilo principal (comportamiento original)
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = frameInput.cols;
            tempCanvas.height = frameInput.rows;
            cv.imshow(tempCanvas, frameInput); // Funciona en el hilo principal
            inputForDetector = tempCanvas;
            // mustDeleteInputForDetector = true; // No se necesita si el canvas es temporal y se descarta
        }
    } else if (frameInput instanceof ImageData) { // Si el worker recibe ImageData directamente
        // Esto puede pasar si el hilo principal envía ImageData, o si se procesa así internamente.
        // En tu caso, alpr_worker.js envía cv.Mat (creado desde ImageData), así que este bloque
        // podría no ser alcanzado con el flujo actual desde el worker.
        matForOcr = cv.matFromImageData(frameInput); // Necesario para OCR
        mustDeleteMatForOcr = true;
        inputForDetector = frameInput; // El detector puede usar ImageData
    } else if (isInWorker && frameInput instanceof OffscreenCanvas) { // Si el worker recibe OffscreenCanvas
        // El detector puede usar OffscreenCanvas.
        inputForDetector = frameInput;
        // Para OCR, necesitamos cv.Mat. Convertir OffscreenCanvas a cv.Mat.
        const ctx = frameInput.getContext('2d');
        if (!ctx) throw new Error("No se pudo obtener el contexto 2D del OffscreenCanvas para OCR.");
        const imgDataFromOffscreen = ctx.getImageData(0, 0, frameInput.width, frameInput.height);
        matForOcr = cv.matFromImageData(imgDataFromOffscreen);
        mustDeleteMatForOcr = true;
    } else if (!isInWorker && frameInput instanceof HTMLCanvasElement) { // Hilo principal y HTMLCanvasElement
        inputForDetector = frameInput;
        matForOcr = cv.imread(inputForDetector); // imread puede tomar canvas
        mustDeleteMatForOcr = true;
    } else if (!isInWorker && frameInput instanceof HTMLImageElement) { // Hilo principal y HTMLImageElement
        inputForDetector = frameInput;
        matForOcr = cv.imread(inputForDetector); // imread puede tomar imagen
        mustDeleteMatForOcr = true;
    }
    else {
        console.error(`Formato de frame no soportado en ALPR.predict: ${frameInput ? frameInput.constructor.name : frameInput}`, "isInWorker:", isInWorker);
        throw new TypeError(`Formato de frame no soportado en ALPR.predict: ${frameInput ? frameInput.constructor.name : frameInput}`);
    }

    // `this.detector.predict` (LicensePlateDetector -> YoloV9ObjectDetector)
    // espera HTMLImageElement, HTMLCanvasElement, ImageData, u OffscreenCanvas.
    // `inputForDetector` ahora debería ser uno de estos (ImageData u OffscreenCanvas en el worker).
    const plateDetections = await this.detector.predict(inputForDetector);
    const out = [];

    try {
        if (plateDetections && plateDetections.length > 0 && matForOcr && !matForOcr.empty()) {
            for (const det of plateDetections) {
                if (!det.boundingBox) {
                    console.warn("Detección sin boundingBox:", det);
                    out.push(new ALPRResult(det, null));
                    continue;
                }
                const { x1, y1, x2, y2 } = det.boundingBox;
                // Validar y ajustar coordenadas para el ROI
                const frameWidth = matForOcr.cols;
                const frameHeight = matForOcr.rows;

                const boundedX1 = Math.max(0, Math.round(x1));
                const boundedY1 = Math.max(0, Math.round(y1));
                const boundedX2 = Math.min(frameWidth, Math.round(x2));
                const boundedY2 = Math.min(frameHeight, Math.round(y2));

                const rectX = boundedX1;
                const rectY = boundedY1;
                const rectWidth = Math.max(0, boundedX2 - boundedX1);
                const rectHeight = Math.max(0, boundedY2 - boundedY1);

                if (rectWidth <= 0 || rectHeight <= 0 || rectX >= frameWidth || rectY >= frameHeight) {
                    console.warn("ALPR: ROI inválido o fuera de límites después de ajuste, saltando OCR:", { x1, y1, x2, y2 }, "ajustado:", { rectX, rectY, rectWidth, rectHeight }, "dims:", { w: frameWidth, h: frameHeight });
                    out.push(new ALPRResult(det, null));
                    continue;
                }

                const roi = matForOcr.roi(new cv.Rect(rectX, rectY, rectWidth, rectHeight));
                let ocrRes = null;
                if (roi && !roi.empty()) {
                    ocrRes = await this.ocr.predict(roi); // this.ocr.predict (DefaultOCR) espera cv.Mat
                    if (!roi.isDeleted()) roi.delete();
                } else {
                    console.warn("ALPR: ROI estaba vacío para la detección:", { rectX, rectY, rectWidth, rectHeight });
                    if (roi && !roi.isDeleted()) roi.delete();
                }
                out.push(new ALPRResult(det, ocrRes));
            }
        } else if (plateDetections && plateDetections.length > 0) {
            plateDetections.forEach(det => out.push(new ALPRResult(det, null)));
            console.warn("ALPR: Se obtuvieron detecciones pero matForOcr no era válido o estaba vacío para OCR.");
        }
        return out;
    } finally {
        // Limpieza de Mats
        if (mustDeleteMatForOcr && matForOcr && !matForOcr.isDeleted()) {
            matForOcr.delete();
        }
        // Si frameInput era el matForOcr original y no se debe borrar aquí (porque vino del worker),
        // se borrará en alpr_worker.js. Aquí solo borramos los que creamos en este método.
        // El mat RGBA temporal debe borrarse si se creó.
        // La variable tempMatRgba solo existe en el scope del if (isInWorker)
        // Para borrarlo aquí, necesitaría estar en un scope más amplio o manejar su borrado dentro del if.
        // El booleano 'mustDeleteTempMat' no está actualmente conectado a una variable 'tempMatRgba' en este scope.
        // Se ha movido el borrado de tempMatRgba dentro de su scope.
    }
  }



  /**
   * Dibuja las detecciones y OCR sobre la imagen de entrada.
   * @param {string|Blob|File|HTMLImageElement|HTMLCanvasElement|cv.Mat} frame
   * @returns {Promise<HTMLCanvasElement>} Canvas con anotaciones
   */
  async drawPredictions(frame) {
    // 1) Crea un canvas con la imagen original
    const srcCanvas = await this.loadImage(frame);

    // 2) Crea un cv.Mat con esa imagen y clónalo para dibujar
    const srcMat  = cv.imread(srcCanvas);
    const drawMat = srcMat.clone();

    try {
      // 3) Usa predict (que Internamente vuelve a loadImage + detector + OCR)
      const results = await this.predict(srcCanvas);

      for (const { detection: det, ocr: ocrRes } of results) {
        const { x1, y1, x2, y2 } = det.boundingBox;

        // Dibujar rectángulo
        cv.rectangle(
          drawMat,
          new cv.Point(x1, y1),
          new cv.Point(x2, y2),
          [36, 255, 12, 255],
          2
        );

        // Dibujar texto OCR si existe
        if (ocrRes?.text) {
          const conf = Array.isArray(ocrRes.confidence)
            ? ocrRes.confidence.reduce((a,b)=>a+b,0)/ocrRes.confidence.length
            : ocrRes.confidence;
          const txt = `${ocrRes.text} ${(conf*100).toFixed(1)}%`;
          cv.putText(
            drawMat,
            txt,
            new cv.Point(x1, y1 - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            [255, 255, 255, 255],
            2
          );
        }

      }
    } finally {
      srcMat.delete();
    }

    // 4) Vuelca el resultado a un nuevo canvas y libéralos
    const outCanvas = document.createElement('canvas');
    cv.imshow(outCanvas, drawMat);
    drawMat.delete();
    return outCanvas;
  }
}

// Exportar para uso con módulos
export {
  ALPR,
  BoundingBox,
  DetectionResult,
  OcrResult,
  BaseDetector,
  BaseOCR,
  DefaultDetector,
  DefaultOCR,
  ALPRResult
};