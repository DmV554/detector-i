// src/alpr/src/alpr.js

// Nota: Se asume que OpenCV (cv) está disponible globalmente.
// En un worker, esto usualmente se logra con importScripts('path/to/opencv.js');
// En el hilo principal, con <script src="path/to/opencv.js"></script>

import {
    LicensePlateDetector,
    BoundingBox as DetectorBoundingBox, // Importar BoundingBox de plateDetector
    DetectionResult as DetectorDetectionResult // Importar DetectionResult de plateDetector
} from '../../detection/plateDetector.js'; // Nueva ruta
import { OnnxOcrRecognizer } from "../../ocr/inference/onnx-ocr-inference.js";

// --- Clases de Datos Específicas de ALPR ---
class OcrResult {
  constructor(text, confidence) {
    this.text = text;
    this.confidence = confidence;
  }
}

class ALPRResult {
  constructor(detection, ocr) {
    this.detection = detection; // Será de tipo DetectorDetectionResult
    this.ocr = ocr;         // Será de tipo OcrResult
  }
}

// --- Clases Base (Abstracciones) ---
class BaseDetector {
  async init(options) { /* Implementar si es necesario para carga de modelos */ }
  async predict(frame) {
    throw new Error("El método predict debe ser implementado por las subclases");
  }
}

class BaseOCR {
  async init(options) { /* Implementar si es necesario para carga de modelos */ }
  async predict(croppedPlate) {
    throw new Error("El método predict debe ser implementado por las subclases");
  }
}

// --- Implementaciones por Defecto ---
 class DefaultDetector extends BaseDetector {
  constructor(options = {}) {
    super();

    const {
        modelName = "yolo-v9-t-384-license-plate-end2end",
        confThresh = 0.4,
        modelsPath = "./models", // Ruta base para los modelos de detección
    } = options;

    // LicensePlateDetector toma un objeto de opciones
    this.detector = new LicensePlateDetector({
        modelsPath: modelsPath,
        detectionModel: modelName,
        confThresh: confThresh,
        // imgSize: [altura, ancho] // Opcional, si se quiere forzar
        // providers: providers // Ya no se pasa aquí directamente
    });
    this.isInitialized = false;
  }

  async init(height,width) {
    if (!this.isInitialized) {
      await this.detector.loadModel(h=height, w=width); // loadModel está en YoloV9ObjectDetector
      this.isInitialized = true;
      console.log("DefaultDetector inicializado.");
    }
  }

  async predict(frame) { // frame: HTMLImageElement, HTMLCanvasElement, ImageData, OffscreenCanvas
    if (!this.isInitialized) {
      throw new Error("DefaultDetector no inicializado. Llama a init() primero.");
    }
    // this.detector.predict devuelve DetectorDetectionResult[]
    // (que ya tiene label, confidence, y DetectorBoundingBox)
    return await this.detector.predict(frame);
  }
}

 class DefaultOCR extends BaseOCR {
  constructor(options = {}) {
    super();
    const {
        hubOcrModel = "global-plates-mobile-vit-v2-model",
        device = "auto", // 'device' podría no ser usado directamente por OnnxOcrRecognizer si usa 'providers'
        providers = null,
        sessOptions = null,
        modelPath = null,
        configPath = null,
        forceDownload = false
    } = options;

    this.ocrModel = new OnnxOcrRecognizer({ // Asumimos que esta es la interfaz correcta
        hubOcrModel: hubOcrModel, // Usado para inicialización
        // device: device, // Verificar si OnnxOcrRecognizer lo usa o solo providers
        providers: providers,
        sessOptions: sessOptions,
        modelPath: modelPath,
        configPath: configPath,
        forceDownload: forceDownload
    });
    this.isInitialized = false;
    this.hubOcrModel = hubOcrModel; // Guardar para init
  }

  async init() {
    if (!this.isInitialized && this.ocrModel && typeof this.ocrModel.initialize === 'function') {
        try {
            await this.ocrModel.initialize(this.hubOcrModel); // Pasa el nombre/identificador del modelo
            this.isInitialized = true;
            console.log("DefaultOCR inicializado.");
        } catch (error) {
            console.error("Error inicializando DefaultOCR:", error);
            this.isInitialized = false; // Asegurar que quede como no inicializado
            throw error; // Re-lanzar para que ALPR.init() pueda capturarlo
        }
    } else if (this.isInitialized) {
        console.log("DefaultOCR ya estaba inicializado.");
    } else {
        console.warn("DefaultOCR: this.ocrModel o this.ocrModel.initialize no están definidos.");
    }
  }

  async predict(croppedPlate) { // Espera cv.Mat
    if (!this.isInitialized) {
      throw new Error("DefaultOCR no inicializado. Llama a init() primero.");
    }
    if (!croppedPlate || croppedPlate.empty()) {
      console.warn("DefaultOCR.predict: croppedPlate está vacío o no es válido.");
      return null;
    }

    let grayPlate = new cv.Mat();
    let ocrResult = null;

    try {
      cv.cvtColor(croppedPlate, grayPlate, cv.COLOR_BGR2GRAY); // Asume BGR, si es RGBA sería COLOR_RGBA2GRAY

      // `run` en OnnxOcrRecognizer puede que no necesite `initialize` en cada llamada
      const res = await this.ocrModel.run(grayPlate, true); // El segundo parámetro `true` podría ser `keepGrayscale`

      let plateTextArray, confidenceObj;
      if (Array.isArray(res) && res.length === 2) {
        [plateTextArray, confidenceObj] = res;
      } else if (res && typeof res === 'object') { // Adaptar si la estructura de 'res' es diferente
        plateTextArray  = res.textArray    || (res.text ? [res.text] : []);
        confidenceObj   = res.probabilities || res.confidence || { mean: () => 0.0 }; // Fallback
      } else {
        console.warn("DefaultOCR: Formato de respuesta de OCR no esperado", res);
        plateTextArray = [];
        confidenceObj = { mean: () => 0.0 };
      }

      const rawText = Array.isArray(plateTextArray) ? (plateTextArray[0] || "") : String(plateTextArray);
      const text = rawText.replace(/_/g, "");
      const meanConf = typeof confidenceObj.mean === 'function' ? confidenceObj.mean() : Number(confidenceObj);

      ocrResult = new OcrResult(text, meanConf);
    } catch (error) {
        console.error("Error en DefaultOCR.predict:", error);
        ocrResult = null; // Devolver null en caso de error
    } finally {
      if (grayPlate && !grayPlate.isDeleted()) grayPlate.delete();
    }
    return ocrResult;
  }
}

// --- Clase Principal ALPR ---
export default class ALPR {
  constructor(options = {}) {
    const {
        detector, // Permite inyectar un detector personalizado
        ocr,      // Permite inyectar un OCR personalizado
        // Opciones para DefaultDetector
        detectorModel = "yolo-v9-t-384-license-plate-end2end",
        detectorConfThresh = 0.4,
        detectorModelsPath = "./models", // Renombrado para claridad
        heightInput = 0,
        widthInput = 0,

        // Opciones para DefaultOCR
        ocrModel = "global-plates-mobile-vit-v2-model",
        ocrDevice = "auto",
        ocrProviders = null, // ['wasm'] o ['webgl'] etc.
        ocrSessOptions = null,
        ocrModelPath = null,
        ocrConfigPath = null,
        ocrForceDownload = false
    } = options;

    this.detector = detector || new DefaultDetector({
        modelName: detectorModel,
        confThresh: detectorConfThresh,
        modelsPath: detectorModelsPath
    });

    this.ocr = ocr || new DefaultOCR({
        hubOcrModel: ocrModel,
        device: ocrDevice, // Considerar si esto es necesario o solo providers
        providers: ocrProviders,
        sessOptions: ocrSessOptions,
        modelPath: ocrModelPath,
        configPath: ocrConfigPath,
        forceDownload: ocrForceDownload
    });
    this.isInitialized = false;
    this.isInWorker = typeof self.document === 'undefined';
    this.heightInput = heightInput;
    this.widthInput = widthInput;
  }

  async init() {
    if (this.isInitialized) {
        console.log("ALPR ya inicializado.");
        return;
    }
    console.log("Inicializando ALPR...");
    await this.detector.init(this.heightInput, this.widthInput);
    await this.ocr.init();
    this.isInitialized = true;
    console.log("ALPR inicializado correctamente.");
  }

  /**
   * Prepara la imagen de entrada para detección y OCR.
   * Devuelve un objeto { inputForDetector, matForOcr, originalWidth, originalHeight }
   * El llamador es responsable de borrar matForOcr si se crea.
   */
  _prepareImageInputs(frameInput) {
    if (!this.isInitialized) {
      throw new Error("ALPR no inicializado. Llama a init() primero.");
    }
    if (!cv || typeof cv.imread === 'undefined') {
      throw new Error("OpenCV (cv) no está disponible o no ha terminado de cargar.");
    }

    let inputForDetector = frameInput; // Lo que YoloV9ObjectDetector espera
    let matForOcr = null;       // cv.Mat para operaciones de OpenCV
    let originalWidth, originalHeight;
    let createdMat = null; // Para rastrear si creamos un Mat que necesita ser borrado

    if (frameInput instanceof cv.Mat) {
        matForOcr = frameInput; // OCR puede usarlo directamente
        originalWidth = matForOcr.cols;
        originalHeight = matForOcr.rows;

        // Convertir cv.Mat a formato para el detector (ImageData o Canvas)
        if (this.isInWorker) {
            let tempMatRgba = new cv.Mat();
            if (matForOcr.channels() === 1) cv.cvtColor(matForOcr, tempMatRgba, cv.COLOR_GRAY2RGBA);
            else if (matForOcr.channels() === 3) cv.cvtColor(matForOcr, tempMatRgba, cv.COLOR_BGR2RGBA);
            else matForOcr.copyTo(tempMatRgba); // Asume RGBA o maneja error

            if (tempMatRgba.empty() || tempMatRgba.type() !== cv.CV_8UC4) {
                console.error("ALPR._prepareImageInputs: Fallo al convertir cv.Mat a RGBA para ImageData.");
                if(!tempMatRgba.isDeleted()) tempMatRgba.delete();
                throw new Error("Fallo al convertir cv.Mat a RGBA para ImageData.");
            }
            inputForDetector = new ImageData(new Uint8ClampedArray(tempMatRgba.data), tempMatRgba.cols, tempMatRgba.rows);
            tempMatRgba.delete(); // El ImageData ahora tiene los datos, tempMatRgba ya no es necesario
        } else { // Hilo principal
            const canvas = document.createElement('canvas');
            cv.imshow(canvas, matForOcr);
            inputForDetector = canvas;
            originalWidth = canvas.width; // Asegurar que las dimensiones sean del canvas
            originalHeight = canvas.height;
        }
    } else if (frameInput instanceof ImageData) {
        inputForDetector = frameInput;
        matForOcr = cv.matFromImageData(frameInput);
        createdMat = matForOcr; // Este mat fue creado aquí
        originalWidth = frameInput.width;
        originalHeight = frameInput.height;
    } else if (frameInput instanceof (this.isInWorker ? OffscreenCanvas : HTMLCanvasElement)) {
        inputForDetector = frameInput; // Detector puede usar Canvas
        const ctx = frameInput.getContext('2d');
        const imgData = ctx.getImageData(0, 0, frameInput.width, frameInput.height);
        matForOcr = cv.matFromImageData(imgData); // Crear Mat desde Canvas para OCR
        createdMat = matForOcr;
        originalWidth = frameInput.width;
        originalHeight = frameInput.height;
    } else if (!this.isInWorker && frameInput instanceof HTMLImageElement) {
        inputForDetector = frameInput; // Detector puede usar HTMLImageElement
        matForOcr = cv.imread(frameInput); // imread crea un nuevo Mat
        createdMat = matForOcr;
        originalWidth = frameInput.naturalWidth;
        originalHeight = frameInput.naturalHeight;
    } else {
        console.error(`Formato de frame no soportado en ALPR: ${frameInput ? frameInput.constructor.name : frameInput}`);
        throw new TypeError(`Formato de frame no soportado en ALPR.`);
    }

    return { inputForDetector, matForOcr, createdMat, originalWidth, originalHeight };
  }


  async predict(frameInput) {
    if (!this.isInitialized) {
        await this.init(); // Intentar inicializar si no lo está
        if(!this.isInitialized) throw new Error("ALPR no pudo inicializarse.");
    }

    let prep = null;
    let results = [];

    try {
        prep = this._prepareImageInputs(frameInput);
        const { inputForDetector, matForOcr, originalWidth, originalHeight } = prep;

        const plateDetections = await this.detector.predict(inputForDetector); // Espera Canvas/ImageData etc.
        results = [];

        if (plateDetections && plateDetections.length > 0 && matForOcr && !matForOcr.empty()) {
            for (const det of plateDetections) { // det es DetectorDetectionResult
                if (!det.boundingBox) {
                    console.warn("Detección sin boundingBox:", det);
                    results.push(new ALPRResult(det, null));
                    continue;
                }

                // Las coordenadas de det.boundingBox son relativas a la imagen original
                const { x1, y1, x2, y2 } = det.boundingBox;

                const boundedX1 = Math.max(0, Math.round(x1));
                const boundedY1 = Math.max(0, Math.round(y1));
                const boundedX2 = Math.min(originalWidth, Math.round(x2));
                const boundedY2 = Math.min(originalHeight, Math.round(y2));

                const rectX = boundedX1;
                const rectY = boundedY1;
                const rectWidth = Math.max(0, boundedX2 - boundedX1);
                const rectHeight = Math.max(0, boundedY2 - boundedY1);

                if (rectWidth <= 0 || rectHeight <= 0 || rectX >= originalWidth || rectY >= originalHeight) {
                    console.warn("ALPR: ROI inválido o fuera de límites:", { rectX, rectY, rectWidth, rectHeight });
                    results.push(new ALPRResult(det, null));
                    continue;
                }

                let roi = null;
                let ocrRes = null;
                try {
                    roi = matForOcr.roi(new cv.Rect(rectX, rectY, rectWidth, rectHeight));
                    if (roi && !roi.empty()) {
                        ocrRes = await this.ocr.predict(roi); // ocr.predict espera cv.Mat
                    } else {
                        console.warn("ALPR: ROI estaba vacío para la detección:", { rectX, rectY, rectWidth, rectHeight });
                    }
                } finally {
                    if (roi && !roi.isDeleted()) roi.delete();
                }
                results.push(new ALPRResult(det, ocrRes));
            }
        } else if (plateDetections && plateDetections.length > 0) {
            // Hay detecciones pero no se pudo procesar con OCR
            plateDetections.forEach(det => results.push(new ALPRResult(det, null)));
            console.warn("ALPR: Se obtuvieron detecciones pero matForOcr no era válido para OCR.");
        }
        return results;

    } catch (error) {
        console.error("Error en ALPR.predict:", error);
        return []; // Devolver array vacío en caso de error mayor
    } finally {
        if (prep && prep.createdMat && !prep.createdMat.isDeleted()) {
            prep.createdMat.delete(); // Borrar el mat principal si lo creamos aquí
        }
        // Si matForOcr era el frameInput original y es un cv.Mat, no lo borramos aquí.
        // El worker que lo envía es responsable de su limpieza.
    }
  }


  async drawPredictions(frameInput, existingResults = null) {
    if (this.isInWorker) {
        console.warn("drawPredictions no está diseñado para usarse directamente en un worker por dependencia del DOM.");
        return null; // O un objeto que represente las anotaciones sin canvas
    }
    if (!cv || typeof cv.imread === 'undefined') {
        throw new Error("OpenCV (cv) no está disponible o no ha terminado de cargar para drawPredictions.");
    }

    let prep = null;
    let srcMatForDrawing = null; // Mat original para obtener dimensiones y dibujar sobre él (o un clon)
    let alprResults = existingResults;

    try {
        // Obtener la imagen como cv.Mat para dibujar y obtener dimensiones
        prep = this._prepareImageInputs(frameInput); // Nos da matForOcr y dimensiones
        srcMatForDrawing = prep.matForOcr; // Este es el Mat que corresponde a frameInput

        if (!srcMatForDrawing || srcMatForDrawing.empty()) {
            console.error("drawPredictions: No se pudo obtener un cv.Mat válido de la entrada.");
            return null;
        }

        if (!alprResults) {
            // Si no se pasan resultados, los calculamos.
            // Pasamos inputForDetector de la preparación para evitar reconversiones innecesarias.
            alprResults = await this.predict(prep.inputForDetector);
        }

        const drawMat = srcMatForDrawing.clone(); // Clonar para no modificar el original si es una referencia externa

        for (const { detection: det, ocr: ocrRes } of alprResults) {
            if (!det || !det.boundingBox) continue;

            const { x1, y1, x2, y2 } = det.boundingBox; // Estas son DetectorBoundingBox
            cv.rectangle(
                drawMat,
                new cv.Point(Math.round(x1), Math.round(y1)),
                new cv.Point(Math.round(x2), Math.round(y2)),
                [36, 255, 12, 255], // Verde BGR_A
                2 // Grosor
            );

            if (ocrRes && ocrRes.text) {
                const conf = typeof ocrRes.confidence === 'number' ? ocrRes.confidence : 0;
                const txt = `${ocrRes.text} (${(conf * 100).toFixed(0)}%)`;
                cv.putText(
                    drawMat,
                    txt,
                    new cv.Point(Math.round(x1), Math.round(y1) - 5), // Posición del texto
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6, // Escala de la fuente
                    [255, 255, 255, 255], // Blanco BGR_A
                    2 // Grosor
                );
            }
        }

        const outCanvas = document.createElement('canvas');
        cv.imshow(outCanvas, drawMat); // Muestra el drawMat en el outCanvas
        if (!drawMat.isDeleted()) drawMat.delete(); // Borra el clon

        return outCanvas;

    } catch (error) {
        console.error("Error en ALPR.drawPredictions:", error);
        return null;
    } finally {
        // Borrar el mat principal que _prepareImageInputs pudo haber creado
        if (prep && prep.createdMat && !prep.createdMat.isDeleted()) {
            prep.createdMat.delete();
        }
        // Si srcMatForDrawing era el frameInput original (cv.Mat), no se borra aquí.
    }
  }
}

// No es necesario exportar BoundingBox y DetectionResult aquí
// si se van a consumir desde plateDetector.js.
// ALPRResult y OcrResult son específicas de este módulo, así que su exportación aquí está bien.

// Exportar para uso con módulos

export {

  ALPR,
  OcrResult,
  BaseDetector,
  BaseOCR,
  DefaultDetector,
  DefaultOCR,
  ALPRResult

};