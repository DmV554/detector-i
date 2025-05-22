// src/ocr/inference/process.js

/**
 * Objeto contenedor para las funciones de procesamiento OCR
 */
export const OCRUtils = {

    /**
     * Lee una fuente de imagen y la convierte a escala de grises usando OpenCV.js.
     * @param {HTMLImageElement | HTMLCanvasElement | ImageData | OffscreenCanvas | cv.Mat} imageSource
     * @returns {cv.Mat | null} La imagen como un objeto cv.Mat en escala de grises.
     */
    loadImageAndConvertToGrayscale: function(imageSource) {
        if (!cv || !cv.imread) {
            throw new Error("OpenCV.js no está listo o cv.imread no se encontró.");
        }

        let mat;
        let mustDeleteInitialMat = false;

        if (imageSource instanceof cv.Mat) {
            mat = imageSource;
        } else if (imageSource instanceof ImageData ||
                   imageSource instanceof OffscreenCanvas ||
                  (typeof HTMLCanvasElement !== 'undefined' && imageSource instanceof HTMLCanvasElement) ||
                  (typeof HTMLImageElement !== 'undefined' && imageSource instanceof HTMLImageElement)
        ) {
            try {
                mat = cv.imread(imageSource);
                mustDeleteInitialMat = true;
            } catch (err) {
                console.error("Error al leer la imagen con cv.imread:", err, imageSource);
                throw new Error(`Fuente de imagen inválida para cv.imread: ${err.message}`);
            }
        } else {
            throw new Error(`Tipo de imageSource no soportado: ${imageSource ? imageSource.constructor.name : imageSource}`);
        }

        if (!mat || mat.empty()) {
             console.warn("cv.imread (o el Mat de entrada) devolvió un Mat vacío.");
             if (mat && mustDeleteInitialMat && !mat.isDeleted()) mat.delete();
             return null;
        }

        let matGray = new cv.Mat();
        if (mat.channels() === 1) {
             matGray = mat.clone();
        } else if (mat.channels() === 3) {
            cv.cvtColor(mat, matGray, cv.COLOR_RGB2GRAY);
        } else if (mat.channels() === 4) {
            cv.cvtColor(mat, matGray, cv.COLOR_RGBA2GRAY);
        } else {
            if (mustDeleteInitialMat && !mat.isDeleted()) mat.delete();
            if (!matGray.isDeleted()) matGray.delete();
            throw new Error(`Formato de canal no soportado: ${mat.channels()}`);
        }

        if (mustDeleteInitialMat && !mat.isDeleted()) {
            mat.delete();
        }
        return matGray; // El llamador es responsable de borrar este matGray
    },

    /**
     * Preprocesa la(s) imagen(es) en escala de grises para el modelo OCR (formato NHWC).
     * @param {cv.Mat[]} grayMats - Un array de imágenes cv.Mat (escala de grises, CV_8UC1).
     * @param {number} targetImgHeight - La altura deseada del modelo OCR.
     * @param {number} targetImgWidth - El ancho deseado del modelo OCR.
     * @param {number} expectedChannels - Canales esperados por el modelo (usualmente 1 para OCR gris).
     * @returns {{data: Uint8Array, shape: number[]}} Objeto con datos y forma [N, H, W, C].
     */
    preprocessOcrInputs: function(grayMats, targetImgHeight, targetImgWidth, expectedChannels = 1) {
        if (!cv || !cv.resize) {
            throw new Error("OpenCV.js no está listo.");
        }
        if (!Array.isArray(grayMats) || grayMats.length === 0) {
            throw new Error("La entrada debe ser un array de cv.Mat no vacío.");
        }

        const batchSize = grayMats.length;
        const H = targetImgHeight;
        const W = targetImgWidth;
        const C = expectedChannels; // Para OCR en escala de grises que espera 1 canal al final.

        // El modelo espera [N, H, W, C]
        const targetShape = [batchSize, H, W, C];
        const tensorData = new Uint8Array(batchSize * H * W * C);
        let tensorOffset = 0; // Offset para escribir en tensorData

        const dsize = new cv.Size(W, H);
        let tempResized = new cv.Mat();
        let finalMatForTensorExtraction = new cv.Mat(); // Mat del cual se extraerán los datos

        try {
            for (let i = 0; i < batchSize; i++) {
                const singleGrayMat = grayMats[i]; // Asumimos que es CV_8UC1
                if (!singleGrayMat || singleGrayMat.empty()) {
                    throw new Error(`Imagen inválida en el índice ${i}: está vacía.`);
                }
                 if (singleGrayMat.channels() !== 1 && C === 1) { // Solo error si esperamos 1 canal pero recibimos más
                    throw new Error(`Imagen en el índice ${i} no es escala de grises de 1 canal, pero se esperan ${C} canales.`);
                }


                // 1. Redimensionar
                cv.resize(singleGrayMat, tempResized, dsize, 0, 0, cv.INTER_LINEAR);

                // 2. Asegurar que tiene `expectedChannels` (C) y es CV_8U.
                // Si C es 1, y tempResized es CV_8UC1, está listo.
                // Si C es 3, y tempResized es CV_8UC1, convertir a CV_8UC3.
                if (C === 1) {
                    if (tempResized.channels() !== 1) {
                        // Convertir a 1 canal si no lo es (ej. si singleGrayMat era color y se pasó por error)
                        cv.cvtColor(tempResized, finalMatForTensorExtraction, cv.COLOR_RGB2GRAY); // o RGBA2GRAY
                    } else if (tempResized.type() !== cv.CV_8UC1) {
                        tempResized.convertTo(finalMatForTensorExtraction, cv.CV_8U);
                    }
                     else {
                        finalMatForTensorExtraction = tempResized; // Usar directamente
                    }
                } else if (C === 3) {
                    if (tempResized.channels() === 1) {
                        cv.cvtColor(tempResized, finalMatForTensorExtraction, cv.COLOR_GRAY2RGB); // o GRAY2BGR
                    } else if (tempResized.channels() === 3 && tempResized.type() !== cv.CV_8UC3) {
                        tempResized.convertTo(finalMatForTensorExtraction, cv.CV_8U);
                    }
                     else if (tempResized.channels() === 4) { // Si es RGBA y queremos RGB
                        cv.cvtColor(tempResized, finalMatForTensorExtraction, cv.COLOR_RGBA2RGB);
                    }
                    else {
                        finalMatForTensorExtraction = tempResized; // Asumir que ya es CV_8UC3
                    }
                } else {
                    throw new Error(`preprocessOcrInputs: expectedChannels=${C} no está soportado si no es 1 o 3.`);
                }


                // 3. Copiar datos. `finalMatForTensorExtraction.data` es un Uint8Array
                // para un Mat CV_8U. Los datos en .data están en orden HWC.
                // Como queremos NHWC y estamos iterando por N (batch), simplemente
                // concatenamos los bloques HWC.
                tensorData.set(finalMatForTensorExtraction.data, tensorOffset);
                tensorOffset += H * W * C;
            }
        } finally {
            if (!tempResized.isDeleted()) tempResized.delete();
            if (finalMatForTensorExtraction !== tempResized && !finalMatForTensorExtraction.isDeleted()) {
                finalMatForTensorExtraction.delete();
            }
        }
        console.debug("Preprocesamiento OCR (NHWC) completado. Shape final del tensor:", targetShape);
        return { data: tensorData, shape: targetShape }; // Shape es [N, H, W, C]
    },



    /**
     * Post-procesa la salida del modelo OCR.
     * @param {Float32Array | Int32Array | Uint8Array} modelOutputData - Salida del modelo.
     * @param {number[]} outputShape - Forma de la salida (ej. [Batch, Timesteps, AlphabetSize]).
     * @param {number} maxPlateSlots - Max slots (debe coincidir con Timesteps).
     * @param {string} modelAlphabet - Alfabeto del modelo.
     * @param {boolean} [returnConfidence=false] - Si retorna confianzas.
     * @returns { [string[], (number[][] | undefined)] } Array de matrículas. Si returnConfidence, [placas, probs].
     */
    postprocessOcrOutput: function(
        modelOutputData,
        outputShape,
        maxPlateSlots, // Este es el 'sequence_length' o 'timesteps'
        modelAlphabet,
        returnConfidence = false
    ) {
        // ... (Tu lógica de postprocessOutput existente era bastante buena)
        // Asegúrate que la lógica de indexación para `modelOutputData`
        // corresponda con el `outputShape` real que tu modelo OCR produce.
        // La decodificación CTC (si es un modelo tipo CRNN) es más compleja que un simple argmax por slot.
        // El ejemplo anterior de CTC decode era un placeholder.
        // Si tu modelo NO es CTC y simplemente predice un carácter por slot:

        if (!Array.isArray(outputShape) || (outputShape.length !== 3 && outputShape.length !== 2)) {
            throw new Error(`outputShape inválido. Se esperaban 2 o 3 dimensiones, se obtuvieron ${outputShape?.length} (${outputShape})`);
        }

        let batchSize, slots, alphabetSizeReal;
        const alphabetArray = modelAlphabet.split('');
        alphabetSizeReal = alphabetArray.length; // Tamaño real del alfabeto que pasaste

        if (outputShape.length === 3) { // [Batch, Slots/Timesteps, NumClasses]
            batchSize = outputShape[0];
            slots = outputShape[1];
            if (outputShape[2] !== alphabetSizeReal) {
                 console.warn(`Tamaño del alfabeto en outputShape (${outputShape[2]}) no coincide con modelAlphabet (${alphabetSizeReal}). Asumiendo ${outputShape[2]}.`);
                 // alphabetSizeReal = outputShape[2]; // Podrías usar el de la forma si confías más en él
            }
        } else { // outputShape.length === 2 asumiendo [Batch, Slots * NumClasses]
            batchSize = outputShape[0];
            const combinedDim = outputShape[1];
            if (combinedDim % alphabetSizeReal !== 0) {
                 throw new Error(`Segunda dimensión (${combinedDim}) no es divisible por tamaño del alfabeto (${alphabetSizeReal})`);
            }
            slots = combinedDim / alphabetSizeReal;
        }

        if (slots !== maxPlateSlots) {
            console.warn(`maxPlateSlots (${maxPlateSlots}) no coincide con slots inferidos/reales de la salida del modelo (${slots}). Usando ${slots}.`);
            maxPlateSlots = slots; // Usar el valor real de la salida
        }

        const expectedLength = batchSize * maxPlateSlots * alphabetSizeReal;
        if (modelOutputData.length !== expectedLength) {
            // Puede que el tensor esté aplanado de forma diferente o sea una forma inesperada.
            console.error("Forma de salida y datos no coinciden:", outputShape, modelOutputData.length, expectedLength);
            throw new Error(`Longitud de modelOutputData (${modelOutputData.length}) no coincide con esperada (${expectedLength}) para forma [${outputShape.join(', ')}] y alfabeto de ${alphabetSizeReal}`);
        }


        const plates = [];
        const probabilitiesList = returnConfidence ? [] : undefined;

        for (let b = 0; b < batchSize; b++) {
            let currentPlate = '';
            const currentProbs = returnConfidence ? [] : undefined;

            for (let s = 0; s < maxPlateSlots; s++) {
                let maxProb = -Infinity;
                let maxIndex = -1;
                const slotOffset = (b * maxPlateSlots + s) * alphabetSizeReal;

                for (let k = 0; k < alphabetSizeReal; k++) {
                    const prob = modelOutputData[slotOffset + k];
                    if (prob > maxProb) {
                        maxProb = prob;
                        maxIndex = k;
                    }
                }

                if (maxIndex !== -1 && maxIndex < alphabetArray.length) { // Asegurar que el índice es válido
                    const char = alphabetArray[maxIndex];
                    // Lógica para manejar el carácter 'blank' o de padding si es un modelo CTC-like o similar
                    // Por ejemplo, si tu 'pad_char' (ej. '_') está en el alfabeto y no debe incluirse:
                    if (char !== '_' ) { // Asumiendo que '_' es el pad_char y no quieres que aparezca.
                                        // O si es un modelo CTC, aquí iría la lógica de colapso.
                        currentPlate += char;
                    }
                    if (returnConfidence && currentProbs) {
                        currentProbs.push(maxProb);
                    }
                } else {
                    // Opcional: manejar el caso de índice inválido, aunque no debería ocurrir.
                    if (returnConfidence && currentProbs) currentProbs.push(0);
                }
            }
            plates.push(currentPlate);
            if (returnConfidence && probabilitiesList && currentProbs) {
                probabilitiesList.push(currentProbs);
            }
        }
        return returnConfidence ? [plates, probabilitiesList] : plates;
    }
};