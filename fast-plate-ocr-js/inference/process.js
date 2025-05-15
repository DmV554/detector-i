

/**
 * Objeto contenedor para las funciones de procesamiento OCR
 */
export const OCRUtils = {

    /**
     * Lee una fuente de imagen y la convierte a escala de grises usando OpenCV.js.
     * Reemplaza a read_plate_image de Python.
     *
     * @param {HTMLImageElement | HTMLCanvasElement | ImageData} imageSource - La fuente de la imagen (ya cargada).
     * @returns {cv.Mat | null} La imagen como un objeto cv.Mat en escala de grises, o null si OpenCV no está listo.
     * @throws {Error} Si la fuente de la imagen no es válida o cv no está inicializado.
     */
    loadImageAndConvertToGrayscale: function(imageSource) {
        if (!cv || !cv.imread) {
            throw new Error("OpenCV.js no está listo o no se encontró.");
        }

        let mat;
        try {
            // cv.imread puede tomar un ID de elemento img/canvas o un elemento directamente
            mat = cv.imread(imageSource);
        } catch (err) {
            console.error("Error al leer la imagen con cv.imread:", err);
            throw new Error("Fuente de imagen inválida para cv.imread.");
        }

        if (mat.empty()) {
             console.warn("cv.imread devolvió un Mat vacío.");
             mat.delete(); // Liberar memoria si se creó un Mat vacío
             return null;
        }

        let matGray = new cv.Mat();
        // Convertir a escala de grises. Asume que la entrada es RGBA o RGB.
        if (mat.channels() === 4) {
            cv.cvtColor(mat, matGray, cv.COLOR_RGBA2GRAY);
        } else if (mat.channels() === 3) {
            cv.cvtColor(mat, matGray, cv.COLOR_RGB2GRAY);
        } else if (mat.channels() === 1) {
             // Ya está en escala de grises (o es un formato inesperado)
             // Clonamos para asegurar que matGray sea independiente
             matGray = mat.clone();
             console.log("La imagen ya parece estar en escala de grises.");
        } else {
            console.error(`Formato de canal no soportado: ${mat.channels()}`);
            mat.delete();
            throw new Error(`Formato de canal no soportado: ${mat.channels()}`);
        }

        mat.delete(); // Liberar memoria de la imagen original a color/rgba
        return matGray;
    },

    /**
     * Preprocesa la(s) imagen(es) en escala de grises para el modelo OCR.
     * Equivalente a preprocess_image de Python.
     *
     * @param {cv.Mat | cv.Mat[]} imageInput - Una imagen cv.Mat (escala de grises) o un array de ellas.
     * @param {number} imgHeight - La altura deseada para el redimensionamiento.
     * @param {number} imgWidth - El ancho deseado para el redimensionamiento.
     * @returns {{data: Float32Array, shape: number[]}} Un objeto con los datos preprocesados como Float32Array plano
     * y la forma [N, H, W, 1].
     * @throws {Error} Si la entrada es inválida o cv no está inicializado.
     */
    preprocessImage: function(imageInput, imgHeight, imgWidth) {
        if (!cv || !cv.resize) {
            throw new Error("OpenCV.js no está listo o no se encontró.");
        }

        const images = Array.isArray(imageInput) ? imageInput : [imageInput]; // Asegurar que sea un array
        if (images.length === 0 || !images[0] || images[0].empty()) {
            throw new Error("La entrada de imagen está vacía o es inválida.");
        }

        const batchSize = images.length;
        const targetShape = [batchSize, imgHeight, imgWidth, 1];
        const totalPixels = batchSize * imgHeight * imgWidth;
        // Los modelos suelen esperar Float32, aunque no normalicemos aquí.
        const processedData = new Uint8Array(totalPixels); // <--- CAMBIAR a Uint8Array

        let offset = 0;
        const dsize = new cv.Size(imgWidth, imgHeight);
        const tempResized = new cv.Mat(); // Reutilizar para eficiencia

        try {
            for (let i = 0; i < batchSize; i++) {
                const img = images[i];
                if (img.empty() || img.channels() !== 1) {
                    console.error(`Imagen inválida en el índice ${i}: ¿está vacía o no es escala de grises? Canales: ${img.channels()}`);
                    continue; // O lanzar un error
                }

                // Redimensionar
                // cv.INTER_LINEAR es la interpolación por defecto en muchos casos, coincide con Python
                cv.resize(img, tempResized, dsize, 0, 0, cv.INTER_LINEAR);

                // Obtener datos y añadirlos al array plano Float32
                const imageData = tempResized.data; // Esto es Uint8Array para CV_8UC1
                for (let j = 0; j < imageData.length; j++) {
                    processedData[offset + j] = imageData[j]; // Copiar valor 0-255 como float
                }
                offset += imageData.length; // Avanzar el offset por el número de píxeles de una imagen
            }
        } finally {
             // Asegurarse de liberar memoria del Mat temporal
            tempResized.delete();
        }


        // Importante: Si usaste cv.Mat que no eran del array original `images`,
        // asegúrate de liberar su memoria (ej. con mat.delete()) si no los necesitas más.

        return { data: processedData, shape: targetShape };
    },

    /**
     * Post-procesa la salida del modelo OCR para obtener las cadenas de texto.
     * Equivalente a postprocess_output de Python.
     *
     * @param {Float32Array | number[]} modelOutputData - Array plano con la salida del modelo.
     * @param {number[]} outputShape - La forma del tensor de salida (ej. [Batch, Timesteps, AlphabetSize]).
     * Se espera que Timesteps sea igual a maxPlateSlots.
     * @param {number} maxPlateSlots - Número máximo de caracteres/slots en la matrícula (debe coincidir con outputShape[1]).
     * @param {string} modelAlphabet - Cadena con los caracteres posibles en el orden esperado por el modelo.
     * @param {boolean} [returnConfidence=false] - Si es true, retorna también las probabilidades máximas.
     * @returns {string[] | [string[], number[][]]} Un array de strings (matrículas decodificadas).
     * Si returnConfidence es true, retorna [plates, probabilities], donde probabilities
     * es un array de arrays con las confianzas para cada slot [N, max_plate_slots].
     * @throws {Error} Si las dimensiones de entrada son inconsistentes.
     */
    postprocessOutput: function(
        modelOutputData,
        outputShape,
        maxPlateSlots,
        modelAlphabet,
        returnConfidence = false
    ) {
        if (!Array.isArray(outputShape) || (outputShape.length !== 3 && outputShape.length !== 2)) {
            // Permitimos [Batch, Slots*Alphabet] o [Batch, Slots, Alphabet]
            throw new Error(`outputShape inválido. Se esperaban 2 o 3 dimensiones, se obtuvieron ${outputShape?.length} (${outputShape})`);
        }


        let batchSize, slots, alphabetSize;
        let expectedLength;

        const alphabetArray = modelAlphabet.split(''); // Convertir string a array de caracteres
        alphabetSize = alphabetArray.length;

        if (outputShape.length === 3) {
             // Formato [Batch, Slots, AlphabetSize]
            batchSize = outputShape[0];
            slots = outputShape[1];
            if (outputShape[2] !== alphabetSize) {
                 throw new Error(`El tamaño del alfabeto en la forma (${outputShape[2]}) no coincide con modelAlphabet (${alphabetSize})`);
            }
            if (slots !== maxPlateSlots) {
                console.warn(`maxPlateSlots (${maxPlateSlots}) no coincide con la dimensión de slots en la forma (${slots}). Usando ${slots}.`);
                maxPlateSlots = slots; // Ajustar al valor real de la salida
            }
            expectedLength = batchSize * slots * alphabetSize;
        } else { // outputShape.length === 2
            // Formato [Batch, Slots * AlphabetSize]
             batchSize = outputShape[0];
             const combinedDim = outputShape[1];
             if (combinedDim % alphabetSize !== 0) {
                 throw new Error(`La segunda dimensión (${combinedDim}) no es divisible por el tamaño del alfabeto (${alphabetSize})`);
             }
             slots = combinedDim / alphabetSize;

              console.log(`Postprocessing - BatchSize: ${batchSize}, Slots: ${slots}, AlphabetSize: ${alphabetSize}`);
    console.log(`Postprocessing - maxPlateSlots recibido: ${maxPlateSlots}`); // Verifica si coincide con 'slots'

              if (slots !== maxPlateSlots) {
                console.warn(`maxPlateSlots (${maxPlateSlots}) calculado de la forma (${slots}) no coincide con el parámetro. Usando ${slots}.`);
                maxPlateSlots = slots; // Ajustar al valor real de la salida
            }
            expectedLength = batchSize * slots * alphabetSize;
        }


        if (modelOutputData.length !== expectedLength) {
            throw new Error(`Longitud de modelOutputData (${modelOutputData.length}) no coincide con la esperada por la forma (${expectedLength}). Shape: [${outputShape.join(', ')}]`);
        }

        const plates = [];
        const probabilities = returnConfidence ? [] : null;

        for (let b = 0; b < batchSize; b++) {
            let currentPlate = '';
            const currentProbs = returnConfidence ? [] : null;

            for (let s = 0; s < maxPlateSlots; s++) {
                let maxProb = -Infinity;
                let maxIndex = -1;

                // Encontrar argmax y max en la dimensión del alfabeto para este slot [b, s]
                for (let k = 0; k < alphabetSize; k++) {
                    // Calcular el índice en el array plano
                    const dataIndex = b * maxPlateSlots * alphabetSize + s * alphabetSize + k;
                    const prob = modelOutputData[dataIndex];

                    if (prob > maxProb) {
                        maxProb = prob;
                        maxIndex = k;
                    }
                }

                if (maxIndex !== -1) {
                    currentPlate += alphabetArray[maxIndex]; // Añadir el caracter decodificado
                    if (returnConfidence) {
                        currentProbs.push(maxProb); // Guardar la probabilidad máxima
                    }
                } else {
                    // Caso improbable si alphabetSize > 0, pero por seguridad
                    currentPlate += '?'; // O algún carácter placeholder
                     if (returnConfidence) {
                        currentProbs.push(0);
                    }
                }
            } // Fin loop slots (s)

            plates.push(currentPlate);
            if (returnConfidence) {
                probabilities.push(currentProbs);
            }
        } // Fin loop batches (b)

        if (returnConfidence) {
            return [plates, probabilities];
        }
        return plates;
    }
};
