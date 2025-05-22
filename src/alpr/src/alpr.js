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
        modelName, // Vendrá de appConfig.detectors[key].fileName
        confThresh,
        modelsPath, // Vendrá de appConfig.paths.detectorModelsBasePath + appConfig.paths.detectorModelsSubPath
        inputHeight,
        inputWidth,
        executionProviders // NUEVO: Recibir EPs
    } = options;

    if (!modelName || !modelsPath) {
        throw new Error("DefaultDetector: modelName y modelsPath son requeridos.");
    }

    this.detector = new LicensePlateDetector({
        modelsPath: modelsPath,
        detectionModel: modelName, // Este es el fileName
        confThresh: confThresh,
        inputHeight: inputHeight,
        inputWidth: inputWidth,
        executionProviders: executionProviders // Pasar EPs a LicensePlateDetector
    });
    this.isInitialized = false;
    this._initialHeight = inputHeight;
    this._initialWidth = inputWidth;
  }

  async init() {
    if (!this.isInitialized) {
      // loadModel en YoloV9ObjectDetector ahora usará los EPs pasados a su constructor
      // y las dimensiones de this.imgSize (establecidas via opciones del constructor de LicensePlateDetector)
      await this.detector.loadModel(this._initialHeight, this._initialWidth); // o solo loadModel() si las dimensiones se manejan enteramente en el constructor del detector
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
             modelPath,    // Ruta completa al .onnx de OCR (de app-config via alpr_worker)
             configPath,   // Ruta completa al JSON de config de OCR (de app-config via alpr_worker)
             executionProviders, // EPs de app-config
             // ocrForceDownload ya no parece relevante si son archivos locales
             // ocrModel (nombre del modelo de app-config, puede usarse para logging o si hay lógica de hub residual)
         } = options;

         if (!modelPath || !configPath) { // Requerir estas rutas
             throw new Error("DefaultOCR: modelPath y configPath son requeridos.");
         }

         this.ocrModel = new OnnxOcrRecognizer({
             modelPath: modelPath,
             configPath: configPath,
             executionProviders: executionProviders, // Pasar los EPs
             // sessionOptions: options.ocrSessOptions // Si tienes más opciones de sesión
         });
         this.isInitialized = false;
         // this.hubOcrModel = options.ocrModel; // Guardar el nombre/key si es útil para logging
     }


     async init() {
         if (!this.isInitialized && this.ocrModel) { // initialize ya no toma argumento
             try {
                 await this.ocrModel.initialize();
                 this.isInitialized = true;
                 console.log("DefaultOCR inicializado (usando OnnxOcrRecognizer con JSON config).");
             } catch (error) {
                 console.error("Error inicializando DefaultOCR:", error);
                 this.isInitialized = false;
                 throw error;
             }
         } else if (this.isInitialized) {
             console.log("DefaultOCR ya estaba inicializado.");
         } else {
             console.warn("DefaultOCR: this.ocrModel o this.ocrModel.initialize no están definidos.");
         }
     }

     async predict(croppedPlate) {
         if (!this.isInitialized) {
             throw new Error("DefaultOCR no inicializado. Llama a init() primero.");
         }
         if (!croppedPlate || croppedPlate.empty()) {
             console.warn("DefaultOCR.predict: croppedPlate está vacío o no es válido.");
             return null;
         }
     // Ya no es necesario convertir a escala de grises aquí,
    // OnnxOcrRecognizer.run lo manejará internamente si es necesario,
    // o OCRUtils.loadImageAndConvertToGrayscale lo hará.
    // Pasamos el croppedPlate directamente.

    let ocrRunResult = null;
    try {
        // OnnxOcrRecognizer.run espera una sola imagen.
        // El segundo parámetro 'returnConfidence' lo he puesto a true en OnnxOcrRecognizer.run
        // para obtener una estructura similar a la que esperabas.
        ocrRunResult = await this.ocrModel.run(croppedPlate, true);

        const text = ocrRunResult && ocrRunResult.textArray && ocrRunResult.textArray.length > 0
                       ? ocrRunResult.textArray[0].replace(/_/g, "") // Tomar el primer texto y limpiar padding
                       : "";
        const meanConf = ocrRunResult && ocrRunResult.probabilities && typeof ocrRunResult.probabilities.mean === 'function'
                       ? ocrRunResult.probabilities.mean()
                       : 0;

        return new OcrResult(text, meanConf);

    } catch (error) {
        console.error("Error en DefaultOCR.predict llamando a ocrModel.run:", error);
        return null;
    }
    // No hay Mats que borrar aquí, OnnxOcrRecognizer maneja los suyos.
    }
 }

export default class ALPR {
  constructor(options = {}) {
    const {
        detector,
        ocr,
        // Detector
        detectorModel,
        detectorConfThresh,
        detectorModelsPath,
        heightInput,
        widthInput,
        detectorExecutionProviders, // NUEVO

        // OCR
        ocrModel,
        ocrModelPath,
        ocrConfigPath,
        ocrForceDownload,
        ocrExecutionProviders // NUEVO
    } = options;

    this.detector = detector || new DefaultDetector({
        modelName: detectorModel,
        confThresh: detectorConfThresh,
        modelsPath: detectorModelsPath,
        inputHeight: heightInput,
        inputWidth: widthInput,
        executionProviders: detectorExecutionProviders // Pasar EPs
    });

    this.ocr = ocr || new DefaultOCR({
        hubOcrModel: ocrModel, // ID para OnnxOcrRecognizer
        modelPath: ocrModelPath,
        configPath: ocrConfigPath,
        forceDownload: ocrForceDownload,
        executionProviders: ocrExecutionProviders // Pasar EPs
    });
    this.isInitialized = false;
    this.isInWorker = typeof self.document === 'undefined';
  }

  async init() {
    if (this.isInitialized) {
        console.log("ALPR ya inicializado.");
        return;
    }
    console.log("Inicializando ALPR...");
    await this.detector.init();
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
    let inputForDetector = frameInput;
    let matForOcr = null;
    let originalWidth, originalHeight;
    let createdMat = null;
    if (frameInput instanceof cv.Mat) {
        matForOcr = frameInput;
        originalWidth = matForOcr.cols;
        originalHeight = matForOcr.rows;
        if (this.isInWorker) {
            let tempMatRgba = new cv.Mat();
            if (matForOcr.channels() === 1) cv.cvtColor(matForOcr, tempMatRgba, cv.COLOR_GRAY2RGBA);
            else if (matForOcr.channels() === 3) cv.cvtColor(matForOcr, tempMatRgba, cv.COLOR_BGR2RGBA);
            else matForOcr.copyTo(tempMatRgba);
            if (tempMatRgba.empty() || tempMatRgba.type() !== cv.CV_8UC4) {
                console.error("ALPR._prepareImageInputs: Fallo al convertir cv.Mat a RGBA para ImageData.");
                if(!tempMatRgba.isDeleted()) tempMatRgba.delete();
                throw new Error("Fallo al convertir cv.Mat a RGBA para ImageData.");
            }
            inputForDetector = new ImageData(new Uint8ClampedArray(tempMatRgba.data), tempMatRgba.cols, tempMatRgba.rows);
            tempMatRgba.delete();
        } else {
            const canvas = document.createElement('canvas');
            cv.imshow(canvas, matForOcr);
            inputForDetector = canvas;
            originalWidth = canvas.width;
            originalHeight = canvas.height;
        }
    } else if (frameInput instanceof ImageData) {
        inputForDetector = frameInput;
        matForOcr = cv.matFromImageData(frameInput);
        createdMat = matForOcr;
        originalWidth = frameInput.width;
        originalHeight = frameInput.height;
    } else if (frameInput instanceof (this.isInWorker ? OffscreenCanvas : HTMLCanvasElement)) {
        inputForDetector = frameInput;
        const ctx = frameInput.getContext('2d');
        const imgData = ctx.getImageData(0, 0, frameInput.width, frameInput.height);
        matForOcr = cv.matFromImageData(imgData);
        createdMat = matForOcr;
        originalWidth = frameInput.width;
        originalHeight = frameInput.height;
    } else if (!this.isInWorker && frameInput instanceof HTMLImageElement) {
        inputForDetector = frameInput;
        matForOcr = cv.imread(frameInput);
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
        await this.init();
        if(!this.isInitialized) throw new Error("ALPR no pudo inicializarse.");
    }
    let prep = null;
    let results = [];
    try {
        prep = this._prepareImageInputs(frameInput);
        const { inputForDetector, matForOcr, originalWidth, originalHeight } = prep;
        const plateDetections = await this.detector.predict(inputForDetector);
        results = [];
        if (plateDetections && plateDetections.length > 0 && matForOcr && !matForOcr.empty()) {
            for (const det of plateDetections) {
                if (!det.boundingBox) {
                    results.push(new ALPRResult(det, null));
                    continue;
                }
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
                    results.push(new ALPRResult(det, null));
                    continue;
                }
                let roi = null;
                let ocrRes = null;
                try {
                    roi = matForOcr.roi(new cv.Rect(rectX, rectY, rectWidth, rectHeight));
                    if (roi && !roi.empty()) {
                        ocrRes = await this.ocr.predict(roi);
                    }
                } finally {
                    if (roi && !roi.isDeleted()) roi.delete();
                }
                results.push(new ALPRResult(det, ocrRes));
            }
        } else if (plateDetections && plateDetections.length > 0) {
            plateDetections.forEach(det => results.push(new ALPRResult(det, null)));
        }
        return results;
    } catch (error) {
        console.error("Error en ALPR.predict:", error);
        return [];
    } finally {
        if (prep && prep.createdMat && !prep.createdMat.isDeleted()) {
            prep.createdMat.delete();
        }
    }
  }

  async drawPredictions(frameInput, existingResults = null) {
    if (this.isInWorker) {
        console.warn("drawPredictions no para worker.");
        return null;
    }
    if (!cv || typeof cv.imread === 'undefined') {
        throw new Error("OpenCV (cv) no disponible para drawPredictions.");
    }
    let prep = null;
    let srcMatForDrawing = null;
    let alprResults = existingResults;
    try {
        prep = this._prepareImageInputs(frameInput);
        srcMatForDrawing = prep.matForOcr;
        if (!srcMatForDrawing || srcMatForDrawing.empty()) {
            console.error("drawPredictions: No se pudo obtener cv.Mat válido.");
            return null;
        }
        if (!alprResults) {
            alprResults = await this.predict(prep.inputForDetector);
        }
        const drawMat = srcMatForDrawing.clone();
        for (const { detection: det, ocr: ocrRes } of alprResults) {
            if (!det || !det.boundingBox) continue;
            const { x1, y1, x2, y2 } = det.boundingBox;
            cv.rectangle(
                drawMat,
                new cv.Point(Math.round(x1), Math.round(y1)),
                new cv.Point(Math.round(x2), Math.round(y2)),
                [36, 255, 12, 255],
                2
            );
            if (ocrRes && ocrRes.text) {
                const conf = typeof ocrRes.confidence === 'number' ? ocrRes.confidence : 0;
                const txt = `${ocrRes.text} (${(conf * 100).toFixed(0)}%)`;
                cv.putText(
                    drawMat,
                    txt,
                    new cv.Point(Math.round(x1), Math.round(y1) - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    [255, 255, 255, 255],
                    2
                );
            }
        }
        const outCanvas = document.createElement('canvas');
        cv.imshow(outCanvas, drawMat);
        if (!drawMat.isDeleted()) drawMat.delete();
        return outCanvas;
    } catch (error) {
        console.error("Error en ALPR.drawPredictions:", error);
        return null;
    } finally {
        if (prep && prep.createdMat && !prep.createdMat.isDeleted()) {
            prep.createdMat.delete();
        }
    }
  }
}

// Exportar las clases que se necesiten externamente
export {
  ALPR, // export default ya lo hace, pero por consistencia con las otras si las necesitaras
  OcrResult,
  BaseDetector,
  BaseOCR,
  DefaultDetector,
  DefaultOCR,
  ALPRResult
};