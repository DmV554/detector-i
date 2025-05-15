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
      // Lee Blob/File como DataURL
      const dataURL = await new Promise((res, rej) => {
        const r = new FileReader();
        r.onload  = () => res(r.result);
        r.onerror = () => rej(new Error('Error leyendo Blob/File'));
        r.readAsDataURL(source);
      });
      imgEl = await loadImgEl(dataURL);

    } else if (source instanceof HTMLImageElement) {
      imgEl = source;

    } else {
      throw new TypeError('Formato de imagen no soportado en loadImage');
    }

    return imgToCanvas(imgEl);
  }


  /**
   * Detecta y reconoce matrículas en una imagen.
   * @param {string|Blob|File|HTMLImageElement|HTMLCanvasElement|cv.Mat} frame
   * @returns {Promise<ALPRResult[]>}
   */
  async predict(frame) {
    // 1) Asegurarnos de trabajar sobre un canvas válido
    const canvas = await this.loadImage(frame);

    // 2) Detección de bounding-boxes con tu detector YOLO
    const plateDetections = await this.detector.predict(canvas);

    // 3) Para OCR: convertir a cv.Mat
    const mat   = cv.imread(canvas);
    const out   = [];

    try {
      for (const det of plateDetections) {
        const { x1, y1, x2, y2 } = det.boundingBox;
        // Ajustar ROI dentro de los límites:
        const rect = new cv.Rect(
          Math.max(x1, 0),
          Math.max(y1, 0),
          Math.min(x2 - x1, mat.cols),
          Math.min(y2 - y1, mat.rows)
        );
        const roi = mat.roi(rect);

        // OCR sobre cada placa
        const ocrRes = await this.ocr.predict(roi);
        out.push(new ALPRResult(det, ocrRes));

        roi.delete();
      }
      return out;
    } finally {
      mat.delete();
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

// Exportar las clases para su uso en el navegador
if (typeof window !== 'undefined') {
  window.FastALPR = {
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