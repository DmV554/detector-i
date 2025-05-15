// detector.js

const ort = window.ort;

/**
 * Carga el modelo ONNX de detección (YOLO).
 * @param {string} modelPath - Ruta al archivo .onnx
 * @returns {Promise<ort.InferenceSession>}
 */
export async function loadDetectorModel(modelPath) {
  // Configurar paths del WASM si lo requieres
  //ort.env.wasm.wasmPaths = './onnx';
  
  const session = await ort.InferenceSession.create(modelPath);
  return session;
}

/**
 * Aplica letterbox a la imagen para mantener la relación de aspecto
 * y rellena con un color de fondo (por defecto [114,114,114]).
 */
function letterbox(image, newShape = [384, 384], color = [114, 114, 114]) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const [h, w] = [image.height, image.width];
  const r = Math.min(newShape[0] / h, newShape[1] / w);

  const newUnpad = [Math.round(w * r), Math.round(h * r)];
  const dw = (newShape[1] - newUnpad[0]) / 2;
  const dh = (newShape[0] - newUnpad[1]) / 2;

  canvas.width = newShape[1];
  canvas.height = newShape[0];
  ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.drawImage(image, dw, dh, newUnpad[0], newUnpad[1]);

  return {
    image: canvas,
    ratio: [r, r],
    padding: [dw, dh],
  };
}

/**
 * Convierte el contenido de una imagen (HTMLImageElement) en el tensor
 * que espera el modelo YOLO. Aplica `letterbox`.
 */
export async function preprocess(imageElement) {
  return new Promise((resolve) => {
    const img = new Image();
    img.src = imageElement.src;
    img.onload = () => {
      const { image, ratio, padding } = letterbox(img);
      const ctx = image.getContext("2d");
      const imgData = ctx.getImageData(0, 0, image.width, image.height);
      const data = imgData.data;

      const float32Array = new Float32Array(image.width * image.height * 3);
      let index = 0;

      // RGBA -> CHW
      for (let i = 0; i < data.length; i += 4) {
        // Normalizado a [0,1]
        float32Array[index] = data[i] / 255; // R
        float32Array[index + image.width * image.height] = data[i + 1] / 255; // G
        float32Array[index + 2 * image.width * image.height] = data[i + 2] / 255; // B
        index++;
      }

      const inputTensor = new ort.Tensor(
        'float32',
        float32Array,
        [1, 3, image.height, image.width]
      );

      resolve({ inputTensor, ratio, padding });
    };
    img.onerror = (err) => {
      console.error("Error cargando la imagen para detección:", err);
      resolve({});
    };
  });
}

/**
 * Convierte la salida cruda de YOLO en un arreglo de detecciones con bounding boxes.
 */
export function convertToDetectionResult(
  predictions,
  classLabels,
  ratio,
  padding,
  scoreThreshold = 0.5
) {
  const results = [];
  // predictions: array con bloques de 7 valores: [0, x1, y1, x2, y2, classId, score]
  for (let i = 0; i < predictions.length; i += 7) {
    const [_, x1, y1, x2, y2, classId, score] = predictions.slice(i, i + 7);
    if (score < scoreThreshold) continue;

    const adjustedX1 = (x1 - padding[0]) / ratio[0];
    const adjustedY1 = (y1 - padding[1]) / ratio[1];
    const adjustedX2 = (x2 - padding[0]) / ratio[0];
    const adjustedY2 = (y2 - padding[1]) / ratio[1];

    results.push({
      label: classLabels[classId] || `Class ${classId}`,
      confidence: score,
      boundingBox: { x1: adjustedX1, y1: adjustedY1, x2: adjustedX2, y2: adjustedY2 },
    });
  }
  return results;
}

/**
 * Corta regiones (bounding boxes) de una imagen original para procesarlas por OCR.
 */
export function cropImage(imageElement, detectedObjects) {
  const img = new Image();
  img.src = imageElement.src;

  const promises = detectedObjects.map((obj) => {
    return new Promise((resolve) => {
      const { x1, y1, x2, y2 } = obj.boundingBox;
      const width = x2 - x1;
      const height = y2 - y1;

      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      canvas.width = width;
      canvas.height = height;

      ctx.drawImage(img, x1, y1, width, height, 0, 0, width, height);

      // Convierto a un DataURL para crear un nuevo objeto Image
      const croppedImageData = canvas.toDataURL("image/png");
      const croppedImg = new Image();
      croppedImg.src = croppedImageData;

      croppedImg.onload = () => resolve(croppedImg);
      croppedImg.onerror = (err) => {
        console.error("Error al cargar imagen recortada:", err);
        resolve(null);
      };
    });
  });

  return promises;
}

/**
 * Función principal para realizar detección en una imagen (HTMLImageElement).
 * Devuelve las detecciones resultantes.
 */
export async function detectPlates(frameImage, modelSession) {
  const { inputTensor, ratio, padding } = await preprocess(frameImage);
  if (!inputTensor) return [];

  // feed al modelo (asumiendo que modelSession corresponde a YOLO)
  const feeds = { [modelSession.inputNames[0]]: inputTensor };
  const output = await modelSession.run(feeds);
  const yoloOutputName = modelSession.outputNames[0]; // Ajustar si tu modelo tiene otro nombre
  const predictions = output[yoloOutputName].data;

  // Suponiendo que solo detectamos clase "Placa"
  const detectedObjects = convertToDetectionResult(predictions, ["Patente"], ratio, padding);

  return detectedObjects;
}
