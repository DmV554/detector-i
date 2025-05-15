// ocr.js

const ort = window.ort;
/**
 * Carga el modelo ONNX de OCR
 * @param {string} modelPath - Ruta al archivo .onnx
 * @returns {Promise<ort.InferenceSession>}
 */
export async function loadOcrModel(modelPath) {
  try {
    //alert("dos --1--");
    
    if (!window.ort) {
      throw new Error("ONNX Runtime Web (ort) no está definido.");
    }

    const session = await ort.InferenceSession.create(modelPath);
    
    //alert("dos --2--");
    return session;
  } catch (error) {
    console.error("Error al cargar el modelo ONNX:", error);
    //alert("Error: " + error.message);
    return null;
  }
}

/**
 * Convierte una imagen a un tensor de escala de grises con tamaño fijo [1, H, W, 1].
 */
function preprocessOCRImage(imageElement, targetWidth, targetHeight) {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = targetWidth;
    canvas.height = targetHeight;

    ctx.drawImage(imageElement, 0, 0, targetWidth, targetHeight);
    
    const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
    const data = imageData.data;
    const grayData = new Uint8Array(targetWidth * targetHeight);

    for (let i = 0; i < data.length; i += 4) {
      // Formula de gris con pesos
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      grayData[i / 4] = gray;
    }

    const inputTensor = new ort.Tensor('uint8', grayData, [1, targetHeight, targetWidth, 1]);
    resolve(inputTensor);
  });
}


/* */
export async function runOCRInference(croppedImageElement, ocrSession, maxPlateSlots) {
  const targetWidth = 140;
  const targetHeight = 70;

  const inputTensor = await preprocessOCRImage(croppedImageElement, targetWidth, targetHeight);
  const feeds = { input: inputTensor };

  const output = await ocrSession.run(feeds);
  const outputKey = Object.keys(output)[0];
  const predictions = output[outputKey].data;

  const alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
  
  let plateChars = "";
  let totalConfidence = 0; // Acumulador de probabilidades
  const confidences = []; // Opcional: Para guardar cada probabilidad individual

  for (let i = 0; i < maxPlateSlots; i++) {
    let bestIndex = 0;
    let bestProb = predictions[i * alphabet.length];

    // Buscar la mejor probabilidad para el carácter actual
    for (let j = 1; j < alphabet.length; j++) {
      const index = i * alphabet.length + j;
      const prob = predictions[index];
      if (prob > bestProb) {
        bestProb = prob;
        bestIndex = j;
      }
    }

    plateChars += alphabet[bestIndex] || "";
    totalConfidence += bestProb; // Sumar al acumulador
    confidences.push(bestProb); // Opcional: Guardar cada valor
  }

  // Calcular promedio (evitar división por cero)
  const averageConfidence = maxPlateSlots > 0 
    ? totalConfidence / maxPlateSlots 
    : 0;

  return {
    plateText: plateChars,
    confidence: averageConfidence,
    // confidences // Opcional: Si quieres devolver cada probabilidad
  };
}

/* */
