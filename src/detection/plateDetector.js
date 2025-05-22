// src/detection/plateDetector.js

// Nota: Se asume que ONNX Runtime (ort) está disponible globalmente.
// En un worker, esto usualmente se logra con importScripts('path/to/onnxruntime.min.js');
// En el hilo principal, con <script src="path/to/onnxruntime.min.js"></script>

/**
 * Base class for bounding box representation
 */
export class BoundingBox {
  constructor(x1, y1, x2, y2) {
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
  }

  get width() {
    return this.x2 - this.x1;
  }

  get height() {
    return this.y2 - this.y1;
  }

  get area() {
    return this.width * this.height;
  }

  get center() {
    const cx = (this.x1 + this.x2) / 2.0;
    const cy = (this.y1 + this.y2) / 2.0;
    return [cx, cy];
  }

  intersection(other) {
    const x1 = Math.max(this.x1, other.x1);
    const y1 = Math.max(this.y1, other.y1);
    const x2 = Math.min(this.x2, other.x2);
    const y2 = Math.min(this.y2, other.y2);

    if (x2 > x1 && y2 > y1) {
      return new BoundingBox(x1, y1, x2, y2);
    }
    return null;
  }

  iou(other) {
    const inter = this.intersection(other);
    if (!inter) {
      return 0.0;
    }
    const interArea = inter.area;
    const unionArea = this.area + other.area - interArea;
    return unionArea > 0 ? interArea / unionArea : 0.0;
  }

  toXywh() {
    return [this.x1, this.y1, this.width, this.height];
  }

  clamp(maxWidth, maxHeight) {
    return new BoundingBox(
      Math.max(0, Math.min(this.x1, maxWidth)),
      Math.max(0, Math.min(this.y1, maxHeight)),
      Math.max(0, Math.min(this.x2, maxWidth)),
      Math.max(0, Math.min(this.y2, maxHeight))
    );
  }

  isValid(frameWidth, frameHeight) {
    return (
      0 <= this.x1 &&
      this.x1 < this.x2 &&
      this.x2 <= frameWidth &&
      0 <= this.y1 &&
      this.y1 < this.y2 &&
      this.y2 <= frameHeight
    );
  }
}

/**
 * Class representing a detection result
 */
export class DetectionResult {
  constructor(label, confidence, boundingBox) {
    this.label = label;
    this.confidence = confidence;
    this.boundingBox = boundingBox;
  }

  static fromDetectionData(bboxData, confidence, classId) {
    const boundingBox = new BoundingBox(...bboxData);
    return new DetectionResult(classId, confidence, boundingBox);
  }
}

/**
 * Utility functions for model preprocessing and postprocessing
 */
export const Utils = {
  letterbox(im, newShape = [640, 640], color = [114, 114, 114], scaleUp = true) {
    let destCanvas, destCtx;
    const isInWorker = typeof self.document === 'undefined';

    if (isInWorker) {
        destCanvas = new OffscreenCanvas(typeof newShape === 'number' ? newShape : newShape[0], typeof newShape === 'number' ? newShape : newShape[1]);
    } else {
        destCanvas = document.createElement('canvas');
    }

    let width, height;
    let imageSourceForDrawing = im;

    if (im instanceof ImageData) {
        width = im.width;
        height = im.height;
        let tempSourceCanvas;
        if (isInWorker) {
            tempSourceCanvas = new OffscreenCanvas(im.width, im.height);
        } else {
            tempSourceCanvas = document.createElement('canvas');
            tempSourceCanvas.width = im.width;
            tempSourceCanvas.height = im.height;
        }
        tempSourceCanvas.getContext('2d').putImageData(im, 0, 0);
        imageSourceForDrawing = tempSourceCanvas;
    } else if (typeof OffscreenCanvas !== 'undefined' && im instanceof OffscreenCanvas) {
        width = im.width;
        height = im.height;
    } else if (!isInWorker) {
        if (typeof HTMLImageElement !== 'undefined' && im instanceof HTMLImageElement) {
            width = im.naturalWidth;
            height = im.naturalHeight;
        } else if (typeof HTMLCanvasElement !== 'undefined' && im instanceof HTMLCanvasElement) {
            width = im.width;
            height = im.height;
        } else {
            console.error("Tipo de imagen no soportado en letterbox (hilo principal, tipo desconocido):", im);
            throw new Error("Tipo de imagen no soportado en letterbox (hilo principal, tipo desconocido)");
        }
    } else {
        console.error("Tipo de imagen no soportado en letterbox (worker, no es ImageData/OffscreenCanvas):", im);
        throw new Error("Tipo de imagen no soportado en letterbox (worker, no es ImageData/OffscreenCanvas)");
    }

    if (typeof width === 'undefined' || width === 0 || typeof height === 'undefined' || height === 0) {
        if (im && typeof im.width === 'number' && typeof im.height === 'number' && im.width > 0 && im.height > 0) {
            width = im.width;
            height = im.height;
        } else {
            console.error("Dimensiones de imagen inválidas o indeterminadas en letterbox. Imagen:", im, "Ancho calc:", width, "Alto calc:", height);
            throw new Error("Dimensiones de imagen inválidas o indeterminadas en letterbox.");
        }
    }

    if (typeof newShape === 'number') {
      newShape = [newShape, newShape];
    }

    if (destCanvas.width !== newShape[0] || destCanvas.height !== newShape[1]) {
        destCanvas.width = newShape[0];
        destCanvas.height = newShape[1];
    }
    destCtx = destCanvas.getContext('2d');

    const r = Math.min(newShape[0] / width, newShape[1] / height);
    const ratio = scaleUp ? r : Math.min(r, 1.0);

    const newUnpad = [Math.round(width * ratio), Math.round(height * ratio)];
    const dw = (newShape[0] - newUnpad[0]) / 2;
    const dh = (newShape[1] - newUnpad[1]) / 2;

    destCtx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    destCtx.fillRect(0, 0, destCanvas.width, destCanvas.height);

    const top = Math.round(dh - 0.1);
    const left = Math.round(dw - 0.1);

    destCtx.drawImage(imageSourceForDrawing, 0, 0, width, height, left, top, newUnpad[0], newUnpad[1]);

    return {
      resizedImage: destCanvas,
      ratio: [ratio, ratio],
      padding: [dw, dh]
    };
  },

  preprocess(img, imgSize) {
    const { resizedImage, ratio, padding } = this.letterbox(img, imgSize);
    const canvasToProcess = resizedImage;
    const ctx = canvasToProcess.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvasToProcess.width, canvasToProcess.height);
    const { data, width, height } = imageData;

    const inputTensor = new Float32Array(1 * 3 * height * width);
    let offset = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelOffset = (y * width + x) * 4;
        inputTensor[offset++] = data[pixelOffset + 0] / 255.0; // R
        inputTensor[offset++] = data[pixelOffset + 1] / 255.0; // G
        inputTensor[offset++] = data[pixelOffset + 2] / 255.0; // B
      }
    }
    // Corrección: el tensor se llena C * H * W, no H * W * C y luego otra vez
    // La lógica original era correcta al indexar inputTensor[channel * H * W + y * W + x]
    // Reimplementando para mayor claridad:
    const redPlane = [];
    const greenPlane = [];
    const bluePlane = [];
    for (let i = 0; i < data.length; i += 4) {
        redPlane.push(data[i] / 255.0);
        greenPlane.push(data[i + 1] / 255.0);
        bluePlane.push(data[i + 2] / 255.0);
    }
    const finalTensor = new Float32Array(1 * 3 * height * width);
    finalTensor.set(redPlane);
    finalTensor.set(greenPlane, redPlane.length);
    finalTensor.set(bluePlane, redPlane.length + greenPlane.length);


    return {
      // tensor: inputTensor, // Original
      tensor: finalTensor, // Corregido
      ratio,
      padding
    };
  },

  convertToDetectionResult(predictions, classLabels, ratio, padding, scoreThreshold = 0.5) {
    const results = [];
    // Assuming predictions shape is [1, N, 7] where N is num_detections
    // and each detection is [batch_idx, x1, y1, x2, y2, class_id, score]
    // The .data flattens it, so we iterate based on the number of elements per detection (7)
    const numDetections = predictions.length / 7;

    for (let i = 0; i < numDetections; i++) {
      const offset = i * 7;
      // const batchIdx = predictions[offset + 0]; // Not typically used client-side for single image
      const x1 = predictions[offset + 1];
      const y1 = predictions[offset + 2];
      const x2 = predictions[offset + 3];
      const y2 = predictions[offset + 4];
      const classId = Math.round(predictions[offset + 5]);
      const score = predictions[offset + 6];

      if (score < scoreThreshold) {
        continue;
      }

      const adjustedBox = [
        Math.round((x1 - padding[0]) / ratio[0]),
        Math.round((y1 - padding[1]) / ratio[1]),
        Math.round((x2 - padding[0]) / ratio[0]),
        Math.round((y2 - padding[1]) / ratio[1])
      ];
      const label = classId < classLabels.length ? classLabels[classId] : classId.toString();
      const boundingBox = new BoundingBox(...adjustedBox);
      const detectionResult = new DetectionResult(label, score, boundingBox);
      results.push(detectionResult);
    }
    return results;
  },

  async measureTime(fn) {
    const start = performance.now();
    const result = await fn();
    const end = performance.now();
    return {
      result,
      timeMs: end - start
    };
  }
};

/**
 * Class implementing YOLO v9 object detection
 */
export class YoloV9ObjectDetector {
  constructor(options) {
    this.modelPath = options.modelPath;
    this.classLabels = options.classLabels;
    this.confThresh = options.confThresh || 0.25;
    // imgSize se establece aquí o en loadModel si se pasan h,w
    this.imgSize = (options.inputHeight && options.inputWidth) ?
                   [options.inputHeight, options.inputWidth] :
                   (options.imgSize || [640, 640]);
    this.executionProviders = options.executionProviders || ['wasm']; // Guardar EPs
    this.modelLoaded = false;
    this.model = null;
    this.inputName = '';
    this.outputName = '';
  }

  async loadModel(h,w) {
    try {
      // 'ort' debería estar disponible globalmente aquí (desde importScripts en worker o <script> en main)
      if (typeof ort === 'undefined') {
        throw new Error("ONNX Runtime (ort) no está disponible globalmente.");
      }

      const sessionOptions = {
        // executionProviders: this.providers, // Se tomará de ort.env si está configurado allí
        graphOptimizationLevel: 'all',
        logSeverityLevel: 2
      };
      // Si this.providers fue explícitamente pasado y se quiere forzar:
      // if (this.providers && this.providers.length > 0) {
      //   sessionOptions.executionProviders = this.providers;
      // }


      console.log(`YoloV9ObjectDetector: Cargando modelo desde ${this.modelPath} con opciones:`, sessionOptions);
      this.model = await ort.InferenceSession.create(this.modelPath, sessionOptions);

      this.inputName = this.model.inputNames[0];
      this.outputName = this.model.outputNames[0];
      console.log(`YoloV9ObjectDetector: Modelo cargado. Entradas: ${this.model.inputNames}, Salidas: ${this.model.outputNames}, Proveedores efectivos: ${this.model.providers}`);


          const sizeMatch = this.modelPath.match(/-(\d{3,4})-/); // ej. yolov9-t-512-model.onnx
          if (sizeMatch && sizeMatch[1]) {
              const parsedSize = parseInt(sizeMatch[1], 10);
              if (!isNaN(parsedSize) && parsedSize > 0) {
                  h = parsedSize;
                  w = parsedSize;
              }
          }
      // }
      this.imgSize = [h, w];


      this.modelLoaded = true;
      console.log(`YoloV9ObjectDetector: Modelo ONNX ${this.modelPath} cargado. Tamaño de entrada inferido/establecido a: ${this.imgSize[0]}x${this.imgSize[1]}`);
    } catch (error) {
      console.error(`YoloV9ObjectDetector: Fallo al cargar modelo ONNX (${this.modelPath}):`, error);
      this.modelLoaded = false; // Asegurar que esté false en caso de error
      throw error;
    }
  }

  async predict(image) {
    if (!this.modelLoaded) {
      throw new Error("YoloV9ObjectDetector: Modelo no cargado. Llama a loadModel() primero.");
    }
    if (typeof ort === 'undefined') {
      throw new Error("ONNX Runtime (ort) no disponible para predicción.");
    }
    let imageToProcess = image;
    const isInWorker = typeof self.document === 'undefined';
    if (typeof image === 'string') {
      if (isInWorker) {
        throw new Error("YoloV9ObjectDetector: Carga desde URL no soportada en worker.");
      }
      imageToProcess = await this._loadImageFromUrl(image);
    }
    const { tensor, ratio, padding } = Utils.preprocess(imageToProcess, this.imgSize);
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, this.imgSize[0], this.imgSize[1]]);
    const feeds = { [this.inputName]: inputTensor };
    try {
      const results = await this.model.run(feeds);
      const outputData = results[this.outputName].data;
      return Utils.convertToDetectionResult(
        outputData,
        this.classLabels,
        ratio,
        padding,
        this.confThresh
      );
    } catch (error) {
      console.error('YoloV9ObjectDetector: Error durante la inferencia:', error);
      return [];
    }
  }

  _loadImageFromUrl(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = (err) => reject(new Error(`Fallo al cargar imagen: ${url} - ${err.toString()}`));
      img.src = url;
    });
  }

  // Método para dibujar predicciones (solo hilo principal)
  async displayPredictions(imageElementOrPath, outputCanvasId) {
    if (typeof self.document === 'undefined') {
        console.warn("displayPredictions no puede usarse en un worker directamente. Se requiere un canvas.");
        return null; // O lanzar error
    }

    let imageToDisplay;
    if (typeof imageElementOrPath === 'string') {
        imageToDisplay = await this._loadImageFromUrl(imageElementOrPath);
    } else {
        imageToDisplay = imageElementOrPath; // Asume HTMLImageElement, HTMLCanvasElement, ImageData
    }

    let canvas;
    if (outputCanvasId) {
        canvas = document.getElementById(outputCanvasId);
        if (!canvas) {
            console.error(`Canvas con ID '${outputCanvasId}' no encontrado.`);
            return null;
        }
    } else {
        canvas = document.createElement('canvas');
    }

    const ctx = canvas.getContext('2d');

    let originalWidth, originalHeight;

    if (imageToDisplay instanceof HTMLImageElement) {
      originalWidth = imageToDisplay.naturalWidth;
      originalHeight = imageToDisplay.naturalHeight;
      canvas.width = originalWidth;
      canvas.height = originalHeight;
      ctx.drawImage(imageToDisplay, 0, 0);
    } else if (imageToDisplay instanceof ImageData) {
      originalWidth = imageToDisplay.width;
      originalHeight = imageToDisplay.height;
      canvas.width = originalWidth;
      canvas.height = originalHeight;
      ctx.putImageData(imageToDisplay, 0, 0);
    } else if (imageToDisplay instanceof HTMLCanvasElement || (typeof OffscreenCanvas !== 'undefined' && imageToDisplay instanceof OffscreenCanvas) ) {
      originalWidth = imageToDisplay.width;
      originalHeight = imageToDisplay.height;
      canvas.width = originalWidth;
      canvas.height = originalHeight;
      ctx.drawImage(imageToDisplay, 0, 0);
    } else {
        console.error("Tipo de imagen no soportado para displayPredictions");
        return null;
    }


    const detections = await this.predict(imageToDisplay); // Reutiliza predict

    for (const detection of detections) {
      const bbox = detection.boundingBox.clamp(originalWidth, originalHeight); // Asegura que el bbox esté dentro de los límites
      const label = `${detection.label}: ${detection.confidence.toFixed(2)}`;

      ctx.strokeStyle = 'rgb(0, 255, 0)';
      ctx.lineWidth = Math.max(1, Math.min(originalWidth, originalHeight) / 200); // Grosor de línea adaptable
      ctx.strokeRect(bbox.x1, bbox.y1, bbox.width, bbox.height);

      const fontSize = Math.max(10, Math.min(originalWidth, originalHeight) / 30);
      ctx.font = `${fontSize}px Arial`;
      const textMetrics = ctx.measureText(label);
      const textWidth = textMetrics.width;
      const textHeight = fontSize * 1.2; // Aproximación de la altura del texto

      ctx.fillStyle = 'rgb(0, 255, 0)';
      ctx.fillRect(bbox.x1, bbox.y1 - textHeight, textWidth + (fontSize * 0.5), textHeight);

      ctx.fillStyle = 'rgb(0, 0, 0)';
      ctx.fillText(label, bbox.x1 + (fontSize * 0.25), bbox.y1 - (textHeight * 0.15));
    }
    return canvas; // Devuelve el canvas (sea el pasado por ID o uno nuevo)
  }
}

/**
 * Nombres de modelos disponibles para la detección de matrículas.
 * Usado para construir la ruta al archivo .onnx.
 * @enum {string}
 */
export const PlateDetectorModel = {
  YOLO_V9_S_608: 'yolo-v9-s-608-license-plates-end2end',
  YOLO_V9_T_640: 'yolo-v9-t-640-license-plates-end2end',
  YOLO_V9_T_512: 'yolo-v9-t-512-license-plates-end2end',
  YOLO_V9_T_416: 'yolo-v9-t-416-license-plates-end2end',
  YOLO_V9_T_384: 'yolo-v9-t-384-license-plates-end2end',
  YOLO_V9_T_256: 'yolo-v9-t-256-license-plates-end2end'
  // Agrega más modelos aquí si es necesario
};

/**
 * Clase específica para detectar matrículas.
 * Extiende YoloV9ObjectDetector con configuración específica para matrículas.
 */
export class LicensePlateDetector extends YoloV9ObjectDetector {
  constructor(options) { // options viene de DefaultDetector
    const modelFileName = options.detectionModel.endsWith('.onnx') ? options.detectionModel : `${options.detectionModel}.onnx`;
    const modelPath = `${options.modelsPath}/${modelFileName}`;

    super({
      modelPath: modelPath,
      classLabels: ['License Plate'],
      confThresh: options.confThresh,
      inputHeight: options.inputHeight, // Asegurar que estas se pasen
      inputWidth: options.inputWidth,   // Asegurar que estas se pasen
      executionProviders: options.executionProviders // Pasar EPs
    });
    console.log(`LicensePlateDetector: Inicializado. Modelo: ${modelPath}, EPs: ${options.executionProviders?.join(', ')}`);
  }
}