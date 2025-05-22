// src/ocr/inference/process.js

/**
 * Objeto contenedor para las funciones de procesamiento OCR
 */
export const OCRUtils = {

    /**
     * Lee una fuente de imagen y la convierte a escala de grises usando OpenCV.js.
     * @param {HTMLImageElement | HTMLCanvasElement | ImageData | OffscreenCanvas | cv.Mat} imageSource
     * @returns {cv.Mat | null} La imagen como un objeto cv.Mat en escala de grises.
     * @throws {Error} Si la fuente de la imagen no es válida o cv no está inicializado.
     */
    loadImageAndConvertToGrayscale: function(imageSource) {
        if (!cv || !cv.imread) {
            throw new Error("OpenCV.js no está listo o cv.imread no se encontró.");
        }

        let mat;
        let mustDeleteInitialMat = false;

        if (imageSource instanceof cv.Mat) {
            mat = imageSource; // El llamador debe decidir si clonar o no.
                           // Para esta función, asumimos que si es un Mat, ya está listo.
        } else if (imageSource instanceof ImageData ||
                   imageSource instanceof OffscreenCanvas ||
                  (typeof HTMLCanvasElement !== 'undefined' && imageSource instanceof HTMLCanvasElement) ||
                  (typeof HTMLImageElement !== 'undefined' && imageSource instanceof HTMLImageElement)
        ) {
            try {
                mat = cv.imread(imageSource); // imread puede manejar varios tipos
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
             matGray = mat.clone(); // Ya está en escala de grises, clonar para no modificar original
        } else if (mat.channels() === 3) {
            cv.cvtColor(mat, matGray, cv.COLOR_RGB2GRAY); // Asume RGB si es canvas/image
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
     * Preprocesa la(s) imagen(es) en escala de grises para el modelo OCR.
     * @param {cv.Mat[]} grayMats - Un array de imágenes cv.Mat (escala de grises).
     * @param {number} targetImgHeight - La altura deseada del modelo OCR.
     * @param {number} targetImgWidth - El ancho deseado del modelo OCR.
     * @param {number} expectedChannels - Canales esperados por el modelo (ej. 1 para gris, 3 para RGB).
     * @returns {{data: Float32Array, shape: number[]}} Objeto con datos y forma [N, C, H, W].
     */
    preprocessOcrInputs: function(grayMats, targetImgHeight, targetImgWidth, expectedChannels = 1) {
        if (!cv || !cv.resize) {
            throw new Error("OpenCV.js no está listo.");
        }
        if (!Array.isArray(grayMats) || grayMats.length === 0) {
            throw new Error("La entrada debe ser un array de cv.Mat no vacío.");
        }

        const batchSize = grayMats.length;
        // El formato de tensor común es NCHW (Batch, Channels, Height, Width)
        const C = expectedChannels;
        const H = targetImgHeight;
        const W = targetImgWidth;
        const targetShape = [batchSize, C, H, W];
        const tensorData = new Float32Array(batchSize * C * H * W);

        let offset = 0;
        const dsize = new cv.Size(W, H);
        let tempResized = new cv.Mat();
        let matToProcess = new cv.Mat();

        try {
            for (let i = 0; i < batchSize; i++) {
                const grayMat = grayMats[i];
                if (!grayMat || grayMat.empty()) {
                    throw new Error(`Imagen inválida en el índice ${i}: está vacía.`);
                }
                if (grayMat.channels() !== 1) {
                     throw new Error(`Imagen en el índice ${i} no está en escala de grises (canales: ${grayMat.channels()}).`);
                }

                cv.resize(grayMat, tempResized, dsize, 0, 0, cv.INTER_LINEAR);

                // Convertir a 'expectedChannels' si es necesario y normalizar
                if (expectedChannels === 1) { // Modelo espera entrada en escala de grises
                    tempResized.convertTo(matToProcess, cv.CV_32F, 1.0 / 255.0); // Normaliza a [0,1]
                } else if (expectedChannels === 3) { // Modelo espera entrada a color (desde gris)
                    let colorMat = new cv.Mat();
                    cv.cvtColor(tempResized, colorMat, cv.COLOR_GRAY2RGB); // o BGR según el modelo
                    colorMat.convertTo(matToProcess, cv.CV_32F, 1.0 / 255.0);
                    colorMat.delete();
                } else {
                    throw new Error(`Número de canales esperados (${expectedChannels}) no soportado para preprocesamiento OCR.`);
                }

                // Copiar datos al tensor en formato NCHW
                const floatData = matToProcess.data32F; // Acceder a los datos Float32
                for (let c = 0; c < C; c++) {
                    for (let h = 0; h < H; h++) {
                        for (let w = 0; w < W; w++) {
                            // Si C=1, el índice es h * W + w
                            // Si C=3, el índice es (h * W + w) * C + c (si los datos están interleaved)
                            // O si OpenCV lo devuelve plano por canal, ajustar.
                            // Asumiendo que matToProcess.data32F está en formato HWC
                            let val;
                            if (C === 1) {
                                val = floatData[h * W + w];
                            } else { // C === 3
                                val = floatData[(h * W + w) * C + c]; // HWC
                            }
                            tensorData[offset + c * (H * W) + h * W + w] = val;
                        }
                    }
                }
                offset += C * H * W; // Avanzar para el siguiente item del batch
            }
        } finally {
            if (!tempResized.isDeleted()) tempResized.delete();
            if (!matToProcess.isDeleted()) matToProcess.delete();
        }
        return { data: tensorData, shape: targetShape };
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