// Contenido MODIFICADO para openimageclaude.js

/**
 * Open Image Models JS
 * A JavaScript port of the Python open-image-models library for license plate detection
 */

// Import ONNX Runtime Web
// Nota: ONNX Runtime ya debería estar disponible globalmente si se carga mediante importScripts en el worker.

/**
 * Base class for bounding box representation
 */
class BoundingBox {
  // ... (código existente de BoundingBox sin cambios) ...
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
class DetectionResult {
   // ... (código existente de DetectionResult sin cambios) ...
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
const Utils = {
  /**
   * Resizes and pads the input image while maintaining aspect ratio
   * @param {HTMLImageElement|ImageData|HTMLCanvasElement|OffscreenCanvas} im - Input image
   * @param {Array<number>|number} newShape - Target shape [width, height] or single number for square
   * @param {Array<number>} color - Padding color [r, g, b]
   * @param {boolean} scaleUp - Whether to scale up the image if it's smaller than target
   * @returns {Object} Resized image (HTMLCanvasElement or OffscreenCanvas), ratio, and padding information
   */
  letterbox(im, newShape = [640, 640], color = [114, 114, 114], scaleUp = true) {
    let destCanvas, destCtx; // Lienzo de destino para la imagen redimensionada y con padding
    const isInWorker = typeof self.document === 'undefined'; // Detección de entorno Worker

    // Crear el lienzo de destino (OffscreenCanvas en worker, HTMLCanvasElement en hilo principal)
    if (isInWorker) {
        destCanvas = new OffscreenCanvas(typeof newShape === 'number' ? newShape : newShape[0], typeof newShape === 'number' ? newShape : newShape[1]);
    } else {
        destCanvas = document.createElement('canvas');
    }

    let width, height;
    let imageSourceForDrawing = im; // Fuente que se usará con ctx.drawImage

    // Determinar dimensiones y preparar la fuente para dibujar, de forma segura para workers
    if (im instanceof ImageData) {
        width = im.width;
        height = im.height;
        // Para dibujar ImageData, primero lo ponemos en un canvas temporal
        let tempSourceCanvas;
        if (isInWorker) {
            tempSourceCanvas = new OffscreenCanvas(im.width, im.height);
        } else {
            tempSourceCanvas = document.createElement('canvas');
            tempSourceCanvas.width = im.width;
            tempSourceCanvas.height = im.height;
        }
        tempSourceCanvas.getContext('2d').putImageData(im, 0, 0);
        imageSourceForDrawing = tempSourceCanvas; // Actualizar la fuente de dibujo
    } else if (typeof OffscreenCanvas !== 'undefined' && im instanceof OffscreenCanvas) {
        // OffscreenCanvas es seguro de comprobar y usar en workers (y main thread si es compatible)
        width = im.width;
        height = im.height;
        // imageSourceForDrawing sigue siendo 'im'
    } else if (!isInWorker) {
        // Estas comprobaciones solo son seguras en el hilo principal (no worker)
        if (typeof HTMLImageElement !== 'undefined' && im instanceof HTMLImageElement) {
            width = im.naturalWidth;
            height = im.naturalHeight;
            // imageSourceForDrawing sigue siendo 'im'
        } else if (typeof HTMLCanvasElement !== 'undefined' && im instanceof HTMLCanvasElement) {
            width = im.width;
            height = im.height;
            // imageSourceForDrawing sigue siendo 'im'
        } else {
            // Si está en el hilo principal y no es ninguno de los tipos esperados
            console.error("Tipo de imagen no soportado en letterbox (hilo principal, tipo desconocido):", im);
            throw new Error("Tipo de imagen no soportado en letterbox (hilo principal, tipo desconocido)");
        }
    } else {
        // Si está en el worker y no es ImageData ni OffscreenCanvas (ya comprobados)
        // Esto no debería ocurrir si ALPR.predict envía ImageData correctamente.
        console.error("Tipo de imagen no soportado en letterbox (worker, no es ImageData/OffscreenCanvas):", im);
        throw new Error("Tipo de imagen no soportado en letterbox (worker, no es ImageData/OffscreenCanvas)");
    }

    // Verificación final de dimensiones (si alguna imagen no cargó bien o tipo inesperado)
    if (typeof width === 'undefined' || width === 0 || typeof height === 'undefined' || height === 0) {
        // Intenta un último recurso si el objeto tiene propiedades width/height (podría ser un objeto genérico similar)
        if (im && typeof im.width === 'number' && typeof im.height === 'number' && im.width > 0 && im.height > 0) {
            width = im.width;
            height = im.height;
        } else {
            console.error("Dimensiones de imagen inválidas o indeterminadas en letterbox. Imagen:", im, "Ancho calc:", width, "Alto calc:", height);
            throw new Error("Dimensiones de imagen inválidas o indeterminadas en letterbox.");
        }
    }

    // Ajustar el tamaño del lienzo de destino si es necesario (newShape puede ser un número o array)
    if (typeof newShape === 'number') {
      newShape = [newShape, newShape];
    }

    if (destCanvas.width !== newShape[0] || destCanvas.height !== newShape[1]) {
        destCanvas.width = newShape[0];
        destCanvas.height = newShape[1];
    }
    destCtx = destCanvas.getContext('2d');

    // Calcular ratio y padding
    const r = Math.min(newShape[0] / width, newShape[1] / height);
    const ratio = scaleUp ? r : Math.min(r, 1.0);

    const newUnpad = [Math.round(width * ratio), Math.round(height * ratio)];
    const dw = (newShape[0] - newUnpad[0]) / 2;
    const dh = (newShape[1] - newUnpad[1]) / 2;

    // Dibujar fondo de padding
    destCtx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    destCtx.fillRect(0, 0, destCanvas.width, destCanvas.height);

    // Calcular posición de la imagen redimensionada
    const top = Math.round(dh - 0.1);
    const left = Math.round(dw - 0.1);

    // Dibujar la imagen fuente (imageSourceForDrawing) en el lienzo de destino
    // 'width' y 'height' aquí son las dimensiones originales de 'im' (antes de redimensionar para newUnpad)
    destCtx.drawImage(imageSourceForDrawing, 0, 0, width, height, left, top, newUnpad[0], newUnpad[1]);

    return {
      resizedImage: destCanvas, // Retorna HTMLCanvasElement (main) u OffscreenCanvas (worker)
      ratio: [ratio, ratio],
      padding: [dw, dh]
    };
  }, // Fin de la función letterbox

  /**
   * Preprocesses an image for model inference
   * @param {HTMLImageElement|ImageData|HTMLCanvasElement|OffscreenCanvas} img - Input image
   * @param {Array<number>|number} imgSize - Target image size
   * @returns {Object} Preprocessed image and metadata
   */
  preprocess(img, imgSize) {
    const { resizedImage, ratio, padding } = this.letterbox(img, imgSize);

    const canvasToProcess = resizedImage; // Esto es ahora un HTMLCanvasElement o un OffscreenCanvas
    const ctx = canvasToProcess.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvasToProcess.width, canvasToProcess.height);
    const { data, width, height } = imageData;

    const inputTensor = new Float32Array(1 * 3 * height * width);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelOffset = (y * width + x) * 4;
        inputTensor[0 * height * width + y * width + x] = data[pixelOffset + 0] / 255.0;
        inputTensor[1 * height * width + y * width + x] = data[pixelOffset + 1] / 255.0;
        inputTensor[2 * height * width + y * width + x] = data[pixelOffset + 2] / 255.0;
      }
    }
    return {
      tensor: inputTensor,
      ratio,
      padding
    };
  },

  // ... (convertToDetectionResult y measureTime sin cambios) ...
  convertToDetectionResult(predictions, classLabels, ratio, padding, scoreThreshold = 0.5) {
    const results = [];
    const numDetections = predictions.length / 7; 

    for (let i = 0; i < numDetections; i++) {
      const offset = i * 7;
      const bbox = [
        predictions[offset + 1], 
        predictions[offset + 2], 
        predictions[offset + 3], 
        predictions[offset + 4]  
      ];
      const classId = Math.round(predictions[offset + 5]);
      const score = predictions[offset + 6];

      if (score < scoreThreshold) {
        continue;
      }

      const adjustedBox = [
        Math.round((bbox[0] - padding[0]) / ratio[0]), 
        Math.round((bbox[1] - padding[1]) / ratio[1]), 
        Math.round((bbox[2] - padding[0]) / ratio[0]), 
        Math.round((bbox[3] - padding[1]) / ratio[1])  
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
class YoloV9ObjectDetector {
  constructor(options) {
    this.modelPath = options.modelPath;
    this.classLabels = options.classLabels;
    this.confThresh = options.confThresh || 0.25;
    this.modelLoaded = false;
    this.model = null;
    this.inputName = '';
    this.outputName = '';
    this.imgSize = [0, 0];
    // Los providers se configuran globalmente en el worker/hilo principal a través de ort.env
  }

  async loadModel() {
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


      let h = 256, w = 256; // Default
      // Lógica para deducir tamaño del modelo si es necesario (ej. de this.modelPath o metadata del modelo)
      // const inputMetadata = this.model.inputs[this.inputName]; // Esto es pseudocódigo, la API real puede variar
      // if (inputMetadata && inputMetadata.dims && inputMetadata.dims.length === 4) {
      //     h = inputMetadata.dims[2];
      //     w = inputMetadata.dims[3];
      // } else {
      //     // Intentar deducir del nombre del archivo
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

  async predict(image) { // image es HTMLImageElement, HTMLCanvasElement, ImageData, OffscreenCanvas
    if (!this.modelLoaded) {
      // Intentar cargar el modelo si aún no está cargado podría ser una opción,
      // pero es mejor asegurar que loadModel() se llame explícitamente durante la inicialización.
      // await this.loadModel(); // Podría causar problemas si se llama concurrentemente.
      throw new Error("YoloV9ObjectDetector: Modelo no cargado. Llama a loadModel() primero.");
    }
     if (typeof ort === 'undefined') { // ort debe estar disponible
        throw new Error("ONNX Runtime (ort) no está disponible globalmente para predicción.");
    }

    // imageToProcess será la imagen ya en un formato que Utils.preprocess puede manejar
    // (ej. OffscreenCanvas si está en worker y la entrada era ImageData)
    let imageToProcess = image;
    const isInWorker = typeof self.document === 'undefined';

    // Si la entrada es un string (URL), _loadImageFromUrl solo funciona en hilo principal.
    // En un worker, se espera que la imagen ya venga como ImageData o similar.
    if (typeof image === 'string') {
      if (!isInWorker) {
        imageToProcess = await this._loadImageFromUrl(image);
      } else {
        throw new Error("YoloV9ObjectDetector: Carga desde URL no soportada en worker. Pasa ImageData/OffscreenCanvas.");
      }
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
    if (typeof self.document === 'undefined') {
        return Promise.reject(new Error("_loadImageFromUrl no puede usarse en un worker."));
    }
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous'; // Para evitar tainted canvas si la URL es de otro origen
      img.onload = () => resolve(img);
      img.onerror = (err) => reject(new Error(`Fallo al cargar imagen desde URL: ${url} - ${err.toString()}`));
      img.src = url;
    });
  }

  // _warmUp, _benchmarkInference, showBenchmark pueden requerir adaptaciones si se usan en worker (ej. cómo crear imagen dummy)
  // displayPredictions usa document.createElement y no debe usarse en el worker.

  /**
   * Displays predictions on an image (SOLO HILO PRINCIPAL)
   * @param {HTMLImageElement|HTMLCanvasElement|ImageData} image - Input image
   * @returns {Promise<HTMLCanvasElement>} Canvas with predictions drawn
   */
  async displayPredictions(image) {
    if (typeof self.document === 'undefined') {
        throw new Error("displayPredictions no puede usarse en un worker debido a la manipulación del DOM.");
    }
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    if (image instanceof HTMLImageElement) {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      ctx.drawImage(image, 0, 0);
    } else if (image instanceof ImageData) {
      canvas.width = image.width;
      canvas.height = image.height;
      ctx.putImageData(image, 0, 0);
    } else { // HTMLCanvasElement
      canvas.width = image.width;
      canvas.height = image.height;
      ctx.drawImage(image, 0, 0);
    }

    const detections = await this.predict(image); // Usa el mismo predict

    for (const detection of detections) {
      const bbox = detection.boundingBox;
      const label = `${detection.label}: ${detection.confidence.toFixed(2)}`;
      ctx.strokeStyle = 'rgb(0, 255, 0)';
      ctx.lineWidth = 2;
      ctx.strokeRect(bbox.x1, bbox.y1, bbox.width, bbox.height);
      const textMetrics = ctx.measureText(label);
      const textWidth = textMetrics.width;
      const textHeight = 20; 
      ctx.fillStyle = 'rgb(0, 255, 0)';
      ctx.fillRect(bbox.x1, bbox.y1 - textHeight, textWidth + 10, textHeight);
      ctx.fillStyle = 'rgb(0, 0, 0)';
      ctx.font = '16px Arial';
      ctx.fillText(label, bbox.x1 + 5, bbox.y1 - 5);
    }
    return canvas;
  }
}

/**
 * Models available for license plate detection
 * @enum {string}
 */
const PlateDetectorModel = {
  YOLO_V9_S_608: 'yolo-v9-s-608-license-plates-end2end',
  YOLO_V9_T_640: 'yolo-v9-t-640-license-plates-end2end',
  YOLO_V9_T_512: 'yolo-v9-t-512-license-plates-end2end',
  YOLO_V9_T_416: 'yolo-v9-t-416-license-plates-end2end',
  YOLO_V9_T_384: 'yolo-v9-t-384-license-plates-end2end',
  YOLO_V9_T_256: 'yolo-v9-t-256-license-plates-end2end'
};

/**
 * Class for detecting license plates in images
 */
// Asegúrate que esta clase también esté disponible/exportada si `alpr.js` la necesita directamente.
// En tu `alpr.js`, importas `LicensePlateDetector` de `../../openimageclaude.js`, así que este es el archivo correcto.
// Re-declaración de LicensePlateDetector para asegurar que usa el YoloV9ObjectDetector modificado
export class LicensePlateDetector extends YoloV9ObjectDetector {
  constructor(options) {
    // La ruta './models' debe ser accesible desde el contexto de ejecución (worker o hilo principal)
    // Si el worker está en la raíz, y 'models' está en la raíz, './models' funciona.
    // Si se necesita más flexibilidad, la ruta base podría pasarse en las opciones.
    const modelsPath = options.modelsPath;
    const modelPath = `${modelsPath}/${options.detectionModel}.onnx`;
    console.log(modelPath)

    super({
      modelPath: modelPath,
      classLabels: ['License Plate'],
      confThresh: options.confThresh || 0.25,
    });
    console.log(`LicensePlateDetector (openimageclaude.js): Inicializado. Modelo: ${modelPath}`);
  }
}

// Exportar clases y utilidades si son usadas por otros módulos importados en el worker
// (como por ALPR.js si este archivo se importa directamente allí)
// Si ALPR.js importa directamente LicensePlateDetector de este archivo, la exportación es necesaria.
// (Confirmado: alpr.js importa `LicensePlateDetector` de `../../openimageclaude.js`)
// No es necesario exportar Utils, BoundingBox, etc., si solo son usados internamente por YoloV9ObjectDetector y LicensePlateDetector.