/**
 * Base module - Define las clases e interfaces básicas.
 */

import { LicensePlateDetector } from "../../openimageclaude.js"
import { OnnxOcrRecognizer } from "../../fast-plate-ocr-js/inference/onnx-ocr-inference.js";

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
   */
  constructor(modelName = "yolo-v9-t-384-license-plate-end2end", confThresh = 0.4, providers = null, sessOptions = null) {
    super();
    // Asumimos que LicensePlateDetector está disponible como una implementación JS externa
    this.detector = new LicensePlateDetector({
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
    ocrForceDownload = false
  } = {}) {
    // Inicializar el detector
    this.detector = detector || new DefaultDetector(
      detectorModel,
      detectorConfThresh,
      detectorProviders,
      detectorSessOptions
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

  async predict(frameInput) { // frameInput será cv.Mat desde el worker
    let inputForDetector; // Será OffscreenCanvas si frameInput es cv.Mat/ImageData en worker
    let matForOcr;        // Será el frameInput (cv.Mat)
    let mustDeleteMatForOcr = false; // Si creamos matForOcr desde algo que no es cv.Mat
    let mustDeleteInputForDetector = false;

    const isInWorker = typeof self.document === 'undefined';

    if (frameInput instanceof cv.Mat) {
        matForOcr = frameInput; // OCR usa el Mat directamente
        // Detector (YOLO) espera Canvas/Image. Convertir Mat a OffscreenCanvas en worker.
        if (isInWorker) {
            inputForDetector = new OffscreenCanvas(frameInput.cols, frameInput.rows);
            cv.imshow(inputForDetector, frameInput); // cv.imshow con OffscreenCanvas
        } else { // Esto no debería ocurrir si el worker envía Mat, pero por si acaso
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = frameInput.cols;
            tempCanvas.height = frameInput.rows;
            cv.imshow(tempCanvas, frameInput);
            inputForDetector = tempCanvas;
            mustDeleteInputForDetector = true; // Si se creó un canvas en hilo ppal
        }
    } else if (frameInput instanceof ImageData) { // Si el worker enviara ImageData
        matForOcr = cv.matFromImageData(frameInput);
        mustDeleteMatForOcr = true;
        if (isInWorker) {
            inputForDetector = new OffscreenCanvas(frameInput.width, frameInput.height);
            inputForDetector.getContext('2d').putImageData(frameInput, 0, 0);
        } else {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = frameInput.width;
            tempCanvas.height = frameInput.height;
            tempCanvas.getContext('2d').putImageData(frameInput, 0, 0);
            inputForDetector = tempCanvas;
            mustDeleteInputForDetector = true;
        }
    } else if (frameInput instanceof (isInWorker ? OffscreenCanvas : HTMLCanvasElement) || frameInput instanceof HTMLImageElement) {
        // Si ya es un canvas/imagen (más probable en hilo principal, o si loadImage lo preparó)
        inputForDetector = frameInput;
        // Crear mat para OCR, si cv está disponible
        if (typeof cv !== 'undefined' && cv.imread) {
            matForOcr = cv.imread(inputForDetector); // imread puede tomar canvas/image
            mustDeleteMatForOcr = true;
        } else {
            throw new Error("OpenCV (cv.imread) no disponible para procesar entrada para OCR.");
        }
    } else {
        throw new TypeError(`Formato de frame no soportado en ALPR.predict: ${frameInput ? frameInput.constructor.name : frameInput}`);
    }

    // Ahora this.detector.predict (que es LicensePlateDetector.predict)
    // recibirá un HTMLCanvasElement (hilo ppal) o un OffscreenCanvas (worker).
    // Y openimageclaude.js fue modificado para manejar OffscreenCanvas.
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
                const rectX = Math.max(0, Math.round(x1));
                const rectY = Math.max(0, Math.round(y1));
                const rectWidth = Math.max(0, Math.round(x2 - rectX)); // Ancho desde x1 ajustado
                const rectHeight = Math.max(0, Math.round(y2 - rectY)); // Alto desde y1 ajustado

                if (rectWidth === 0 || rectHeight === 0 ||
                    rectX + rectWidth > matForOcr.cols ||
                    rectY + rectHeight > matForOcr.rows) {
                    console.warn("ALPR: ROI inválido o fuera de límites, saltando OCR para esta detección:", {x1,y1,x2,y2}, "dims:", {w:matForOcr.cols, h:matForOcr.rows});
                    out.push(new ALPRResult(det, null));
                    continue;
                }

                const roi = matForOcr.roi(new cv.Rect(rectX, rectY, rectWidth, rectHeight));
                let ocrRes = null;
                if (!roi.empty()) {
                    ocrRes = await this.ocr.predict(roi); // this.ocr.predict (DefaultOCR) espera cv.Mat
                    if (!roi.isDeleted()) roi.delete(); // Asegurar que ROI se borre
                } else {
                    console.warn("ALPR: ROI estaba vacío para la detección:", {x1,y1,x2,y2});
                    if (!roi.isDeleted()) roi.delete(); // Borrar aunque esté vacío
                }
                out.push(new ALPRResult(det, ocrRes));
            }
        } else if (plateDetections && plateDetections.length > 0) {
            // Hay detecciones pero no se pudo procesar matForOcr
             plateDetections.forEach(det => out.push(new ALPRResult(det, null)));
             console.warn("ALPR: Se obtuvieron detecciones pero matForOcr no era válido para OCR.");
        }
        return out;
    } finally {
        if (mustDeleteMatForOcr && matForOcr && !matForOcr.isDeleted()) {
            matForOcr.delete();
        }
        // No borrar inputForDetector aquí si es el mismo que frameInput (cv.Mat)
        // o si es un OffscreenCanvas que podría ser reutilizado o manejado por el llamador.
        // La gestión de memoria de OffscreenCanvas es menos explícita que cv.Mat.
        if (mustDeleteInputForDetector && inputForDetector && inputForDetector instanceof HTMLCanvasElement && !isInWorker) {
             // inputForDetector.remove(); // Solo si fue un elemento temporal del DOM
        }
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