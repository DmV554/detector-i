/**
 * Open Image Models JS
 * A JavaScript port of the Python open-image-models library for license plate detection
 */

// Import ONNX Runtime Web
// Note: You'll need to add script tag for ONNX Runtime Web in your HTML
// <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>

/**
 * Base class for bounding box representation
 */
class BoundingBox {
  /**
   * Creates a new bounding box
   * @param {number} x1 - X-coordinate of the top-left corner
   * @param {number} y1 - Y-coordinate of the top-left corner
   * @param {number} x2 - X-coordinate of the bottom-right corner
   * @param {number} y2 - Y-coordinate of the bottom-right corner
   */
  constructor(x1, y1, x2, y2) {
    this.x1 = x1;
    this.y1 = y1;
    this.x2 = x2;
    this.y2 = y2;
  }

  /**
   * Gets the width of the bounding box
   * @returns {number} Width of the bounding box
   */
  get width() {
    return this.x2 - this.x1;
  }

  /**
   * Gets the height of the bounding box
   * @returns {number} Height of the bounding box
   */
  get height() {
    return this.y2 - this.y1;
  }

  /**
   * Gets the area of the bounding box
   * @returns {number} Area of the bounding box
   */
  get area() {
    return this.width * this.height;
  }

  /**
   * Gets the center of the bounding box
   * @returns {Array<number>} [x, y] coordinates of the center
   */
  get center() {
    const cx = (this.x1 + this.x2) / 2.0;
    const cy = (this.y1 + this.y2) / 2.0;
    return [cx, cy];
  }

  /**
   * Computes the intersection with another bounding box
   * @param {BoundingBox} other - Another bounding box
   * @returns {BoundingBox|null} Intersection bounding box or null if no intersection
   */
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

  /**
   * Computes the Intersection over Union (IoU) with another bounding box
   * @param {BoundingBox} other - Another bounding box
   * @returns {number} IoU value between 0 and 1
   */
  iou(other) {
    const inter = this.intersection(other);

    if (!inter) {
      return 0.0;
    }

    const interArea = inter.area;
    const unionArea = this.area + other.area - interArea;
    return unionArea > 0 ? interArea / unionArea : 0.0;
  }

  /**
   * Converts bounding box to (x, y, width, height) format
   * @returns {Array<number>} [x, y, width, height]
   */
  toXywh() {
    return [this.x1, this.y1, this.width, this.height];
  }

  /**
   * Clamps the bounding box coordinates to maximum width and height
   * @param {number} maxWidth - Maximum width
   * @param {number} maxHeight - Maximum height
   * @returns {BoundingBox} Clamped bounding box
   */
  clamp(maxWidth, maxHeight) {
    return new BoundingBox(
      Math.max(0, Math.min(this.x1, maxWidth)),
      Math.max(0, Math.min(this.y1, maxHeight)),
      Math.max(0, Math.min(this.x2, maxWidth)),
      Math.max(0, Math.min(this.y2, maxHeight))
    );
  }

  /**
   * Checks if the bounding box is valid
   * @param {number} frameWidth - Frame width
   * @param {number} frameHeight - Frame height
   * @returns {boolean} True if valid, false otherwise
   */
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
  /**
   * Creates a new detection result
   * @param {string} label - Detected object label
   * @param {number} confidence - Confidence score of the detection
   * @param {BoundingBox} boundingBox - Bounding box of the detected object
   */
  constructor(label, confidence, boundingBox) {
    this.label = label;
    this.confidence = confidence;
    this.boundingBox = boundingBox;
  }

  /**
   * Creates a DetectionResult from raw detection data
   * @param {Array<number>} bboxData - Bounding box coordinates [x1, y1, x2, y2]
   * @param {number} confidence - Confidence score
   * @param {string} classId - Class label
   * @returns {DetectionResult} New detection result
   */
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
   * @param {HTMLImageElement|ImageData|HTMLCanvasElement} im - Input image
   * @param {Array<number>|number} newShape - Target shape [width, height] or single number for square
   * @param {Array<number>} color - Padding color [r, g, b]
   * @param {boolean} scaleUp - Whether to scale up the image if it's smaller than target
   * @returns {Object} Resized image, ratio, and padding information
   */
  letterbox(im, newShape = [640, 640], color = [114, 114, 114], scaleUp = true) {
    // Create a canvas to work with the image
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Handle different input types to get width and height
    let width, height;
    if (im instanceof HTMLImageElement) {
      width = im.naturalWidth;
      height = im.naturalHeight;
    } else if (im instanceof ImageData) {
      width = im.width;
      height = im.height;
    } else {
      width = im.width;
      height = im.height;
    }

    // Convert integer newShape to a tuple (newShape, newShape)
    if (typeof newShape === 'number') {
      newShape = [newShape, newShape];
    }

    // Calculate the scaling ratio
    const r = Math.min(newShape[0] / width, newShape[1] / height);
    const ratio = scaleUp ? r : Math.min(r, 1.0);

    // Calculate new unpadded dimensions and padding
    const newUnpad = [Math.round(width * ratio), Math.round(height * ratio)];
    const dw = (newShape[0] - newUnpad[0]) / 2;
    const dh = (newShape[1] - newUnpad[1]) / 2;

    // Set canvas dimensions to target size
    canvas.width = newShape[0];
    canvas.height = newShape[1];

    // Fill with padding color
    ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw the resized image in the center
    const top = Math.round(dh - 0.1);
    const left = Math.round(dw - 0.1);

    if (im instanceof HTMLImageElement) {
      ctx.drawImage(im, 0, 0, width, height, left, top, newUnpad[0], newUnpad[1]);
    } else {
      // For ImageData or canvas, first draw to intermediate canvas
      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = width;
      tmpCanvas.height = height;
      const tmpCtx = tmpCanvas.getContext('2d');

      if (im instanceof ImageData) {
        tmpCtx.putImageData(im, 0, 0);
      } else {
        tmpCtx.drawImage(im, 0, 0);
      }

      ctx.drawImage(tmpCanvas, 0, 0, width, height, left, top, newUnpad[0], newUnpad[1]);
    }

    return {
      resizedImage: canvas,
      ratio: [ratio, ratio],
      padding: [dw, dh]
    };
  },

  /**
   * Preprocesses an image for model inference
   * @param {HTMLImageElement|ImageData|HTMLCanvasElement} img - Input image
   * @param {Array<number>|number} imgSize - Target image size
   * @returns {Object} Preprocessed image and metadata
   */
  preprocess(img, imgSize) {
    // Resize and pad the image
    const { resizedImage, ratio, padding } = this.letterbox(img, imgSize);

    // Get image data from canvas
    const canvas = resizedImage;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const { data, width, height } = imageData;

    // Create a Float32Array for the model input
    // ONNX models expect NCHW format (batch, channels, height, width)
    const inputTensor = new Float32Array(1 * 3 * height * width);

    // Convert from RGBA to RGB and from [0, 255] to [0, 1]
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelOffset = (y * width + x) * 4;

        // RGB channels in CHW format
        // R channel
        inputTensor[0 * height * width + y * width + x] = data[pixelOffset + 0] / 255.0;
        // G channel
        inputTensor[1 * height * width + y * width + x] = data[pixelOffset + 1] / 255.0;
        // B channel
        inputTensor[2 * height * width + y * width + x] = data[pixelOffset + 2] / 255.0;
      }
    }

    return {
      tensor: inputTensor,
      ratio,
      padding
    };
  },

  /**
   * Converts raw model predictions to DetectionResult objects
   * @param {Float32Array} predictions - Model predictions
   * @param {Array<string>} classLabels - Class labels
   * @param {Array<number>} ratio - Scaling ratio [rx, ry]
   * @param {Array<number>} padding - Padding values [dw, dh]
   * @param {number} scoreThreshold - Confidence threshold
   * @returns {Array<DetectionResult>} List of detection results
   */
  convertToDetectionResult(predictions, classLabels, ratio, padding, scoreThreshold = 0.5) {
    const results = [];
    const numDetections = predictions.length / 7; // 7 values per detection

    for (let i = 0; i < numDetections; i++) {
      const offset = i * 7;

      // Extract box coordinates (x1, y1, x2, y2)
      const bbox = [
        predictions[offset + 1], // x1
        predictions[offset + 2], // y1
        predictions[offset + 3], // x2
        predictions[offset + 4]  // y2
      ];

      // Class ID
      const classId = Math.round(predictions[offset + 5]);

      // Confidence score
      const score = predictions[offset + 6];

      // Skip detections below the threshold
      if (score < scoreThreshold) {
        continue;
      }

      // Adjust bounding box to original image size
      const adjustedBox = [
        Math.round((bbox[0] - padding[0]) / ratio[0]), // x1
        Math.round((bbox[1] - padding[1]) / ratio[1]), // y1
        Math.round((bbox[2] - padding[0]) / ratio[0]), // x2
        Math.round((bbox[3] - padding[1]) / ratio[1])  // y2
      ];

      // Map class ID to label
      const label = classId < classLabels.length ? classLabels[classId] : classId.toString();

      // Create bounding box and detection result
      const boundingBox = new BoundingBox(...adjustedBox);
      const detectionResult = new DetectionResult(label, score, boundingBox);

      results.push(detectionResult);
    }

    return results;
  },

  /**
   * Measures execution time of a function
   * @param {Function} fn - Function to measure
   * @returns {Promise<Object>} Object with result and time in milliseconds
   */
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
  /**
   * Creates a new YOLO v9 object detector
   * @param {Object} options - Configuration options
   * @param {string} options.modelPath - Path to the ONNX model file
   * @param {Array<string>} options.classLabels - List of class labels
   * @param {number} options.confThresh - Confidence threshold
   * @param {Array<string>} options.providers - ONNX Runtime providers
   */
  constructor(options) {
    this.modelPath = options.modelPath;
    this.classLabels = options.classLabels;
    this.confThresh = options.confThresh || 0.25;
    this.providers = options.providers || ['wasm'];
    this.modelLoaded = false;
    this.model = null;
    this.inputName = '';
    this.outputName = '';
    this.imgSize = [0, 0];
  }

  /**
   * Loads the ONNX model
   * @returns {Promise<void>}
   */
  async loadModel() {
    try {
      // Create session options
      const sessionOptions = {
        executionProviders: this.providers,
        graphOptimizationLevel: 'all'
      };

      // Create inference session
      this.model = await ort.InferenceSession.create(this.modelPath, sessionOptions);

      // Get input and output names
      this.inputName = this.model.inputNames[0];
      this.outputName = this.model.outputNames[0];

      // Extract height and width (assuming NCHW format)
      const h = 512
      const w = 512

      if (h !== w) {
        throw new Error(`Model only supports square images, but received shape: ${h}x${w}`);
      }

      this.imgSize = [h, w];
      this.modelLoaded = true;

      console.log(`Model loaded successfully with input size: ${h}x${w}`);
    } catch (error) {
      console.error('Failed to load ONNX model:', error);
      throw error;
    }
  }

  /**
   * Performs object detection on an image
   * @param {HTMLImageElement|HTMLCanvasElement|ImageData|string} image - Input image or image URL
   * @returns {Promise<Array<DetectionResult>>} List of detection results
   */
  async predict(image) {
    if (!this.modelLoaded) {
      await this.loadModel();
    }

    // Load image if it's a URL
    if (typeof image === 'string') {
      image = await this._loadImageFromUrl(image);
    }

    // Process the image
    const { tensor, ratio, padding } = Utils.preprocess(image, this.imgSize);

    // Create ONNX tensor
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, this.imgSize[0], this.imgSize[1]]);
    const feeds = {};
    feeds[this.inputName] = inputTensor;

    // Run inference
    try {
      const results = await this.model.run(feeds);
      const outputData = results[this.outputName].data;

      // Convert predictions to detection results
      return Utils.convertToDetectionResult(
        outputData,
        this.classLabels,
        ratio,
        padding,
        this.confThresh
      );
    } catch (error) {
      console.error('Error during model inference:', error);
      return [];
    }
  }

  /**
   * Loads an image from a URL
   * @param {string} url - Image URL
   * @returns {Promise<HTMLImageElement>} Loaded image
   * @private
   */
  _loadImageFromUrl(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = (err) => reject(new Error(`Failed to load image: ${err}`));
      img.src = url;
    });
  }

  /**
   * Performs a warmup run
   * @param {HTMLImageElement} image - Image for warmup
   * @param {number} numRuns - Number of warmup runs
   * @returns {Promise<void>}
   */
  async _warmUp(image, numRuns = 10) {
    console.log(`Starting model warm-up with ${numRuns} runs...`);
    for (let i = 0; i < numRuns; i++) {
      await this.predict(image);
    }
    console.log('Model warm-up completed.');
  }

  /**
   * Benchmarks the model performance
   * @param {HTMLImageElement} image - Image for benchmarking
   * @param {number} numRuns - Number of benchmark runs
   * @returns {Promise<number>} Total time in milliseconds
   */
  async _benchmarkInference(image, numRuns) {
    console.log(`Starting benchmark with ${numRuns} runs...`);
    let totalTimeMs = 0;

    for (let i = 0; i < numRuns; i++) {
      const { timeMs } = await Utils.measureTime(async () => {
        return await this.predict(image);
      });
      totalTimeMs += timeMs;
    }

    console.log('Benchmark completed.');
    return totalTimeMs;
  }

  /**
   * Shows benchmark results
   * @param {number} numRuns - Number of benchmark runs
   * @returns {Promise<void>}
   */
  async showBenchmark(numRuns = 1000) {
    // Create a test image (blank canvas of the right size)
    const canvas = document.createElement('canvas');
    canvas.width = this.imgSize[0];
    canvas.height = this.imgSize[1];
    const ctx = canvas.getContext('2d');

    // Fill with random pixel values
    const imgData = ctx.createImageData(canvas.width, canvas.height);
    for (let i = 0; i < imgData.data.length; i += 4) {
      imgData.data[i + 0] = Math.floor(Math.random() * 256); // R
      imgData.data[i + 1] = Math.floor(Math.random() * 256); // G
      imgData.data[i + 2] = Math.floor(Math.random() * 256); // B
      imgData.data[i + 3] = 255; // A
    }
    ctx.putImageData(imgData, 0, 0);

    // Warm up
    await this._warmUp(canvas, 100);

    // Benchmark
    const totalTimeMs = await this._benchmarkInference(canvas, numRuns);
    const avgTimeMs = totalTimeMs / numRuns;
    const fps = 1000 / avgTimeMs;

    // Display results
    console.log('YoloV9 Object Detector Performance');
    console.log('==================================');
    console.log(`Model: ${this.modelPath}`);
    console.log(`Provider: ${this.providers.join(', ')}`);
    console.log(`Number of Runs: ${numRuns}`);
    console.log(`Average Time (ms): ${avgTimeMs.toFixed(2)}`);
    console.log(`Frames Per Second (FPS): ${fps.toFixed(2)}`);

    return {
      modelPath: this.modelPath,
      providers: this.providers,
      numRuns,
      avgTimeMs,
      fps
    };
  }

  /**
   * Displays predictions on an image
   * @param {HTMLImageElement|HTMLCanvasElement|ImageData} image - Input image
   * @returns {Promise<HTMLCanvasElement>} Canvas with predictions drawn
   */
  async displayPredictions(image) {
    // Create a canvas to work with
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Set canvas dimensions
    if (image instanceof HTMLImageElement) {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      ctx.drawImage(image, 0, 0);
    } else if (image instanceof ImageData) {
      canvas.width = image.width;
      canvas.height = image.height;
      ctx.putImageData(image, 0, 0);
    } else {
      canvas.width = image.width;
      canvas.height = image.height;
      ctx.drawImage(image, 0, 0);
    }

    // Get predictions
    const detections = await this.predict(image);

    // Draw predictions
    for (const detection of detections) {
      const bbox = detection.boundingBox;
      const label = `${detection.label}: ${detection.confidence.toFixed(2)}`;

      // Draw bounding box
      ctx.strokeStyle = 'rgb(0, 255, 0)';
      ctx.lineWidth = 2;
      ctx.strokeRect(bbox.x1, bbox.y1, bbox.width, bbox.height);

      // Draw label background
      const textMetrics = ctx.measureText(label);
      const textWidth = textMetrics.width;
      const textHeight = 20; // Approximate height of the text

      ctx.fillStyle = 'rgb(0, 255, 0)';
      ctx.fillRect(bbox.x1, bbox.y1 - textHeight, textWidth + 10, textHeight);

      // Draw label text
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
export class LicensePlateDetector extends YoloV9ObjectDetector {
  /**
   * Creates a new license plate detector
   * @param {Object} options - Configuration options
   * @param {string} options.detectionModel - Which model to use (see PlateDetectorModel)
   * @param {string} options.modelsPath - Path to the directory containing models
   * @param {number} options.confThresh - Confidence threshold
   * @param {Array<string>} options.providers - ONNX Runtime providers
   */
  constructor(options) {
    // Set default models path if not provided
    const modelsPath = options.modelsPath || './models';

    // Create the model path based on the selected model
    const modelPath = `${modelsPath}/${options.detectionModel}.onnx`;

    // Initialize with superclass
    super({
      modelPath: modelPath,
      classLabels: ['License Plate'],
      confThresh: options.confThresh || 0.25,
      providers: options.providers || ['wasm']
    });

    console.log(`Initialized LicensePlateDetector with model ${modelPath}`);
  }
}

