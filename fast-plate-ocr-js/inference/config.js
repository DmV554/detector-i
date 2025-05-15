/**
 * @fileoverview Configuración del Modelo OCR.
 * Reemplaza la funcionalidad de config.py definiendo la configuración
 * directamente en objetos JavaScript, evitando la necesidad de parsear YAML.
 */

/**
 * @typedef {object} PlateOcrConfig
 * Parámetros de configuración necesarios para el preprocesamiento y
 * postprocesamiento del modelo OCR. Equivalente al PlateOCRConfig TypedDict en Python.
 *
 * @property {number} max_plate_slots - Número máximo de "espacios" o caracteres que predice el modelo.
 * @property {string} alphabet - Cadena que contiene todos los caracteres posibles que el modelo puede emitir, en orden.
 * @property {string} pad_char - Carácter de relleno. Nota: A menudo no se usa directamente en el postprocesamiento JS si se maneja el final de la secuencia de otra manera, pero lo mantenemos por equivalencia.
 * @property {number} img_height - Altura de imagen objetivo para la entrada del modelo.
 * @property {number} img_width - Ancho de imagen objetivo para la entrada del modelo.
 */

/**
 * Configuración para un modelo OCR específico (ej. 'ocr-latin-v1').
 *
 * TODO: Reemplaza estos valores de ejemplo con la configuración real
 * correspondiente a tu modelo OCR cargado desde hub-ocr.js.
 *
 * @type {PlateOcrConfig}
 */
export const OCR_DEFAULT_CONFIG = {
    // --- VALORES DE EJEMPLO - AJUSTAR A TU MODELO ---
    max_plate_slots: 9,       // Ejemplo: Máximo 9 caracteres
    alphabet: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_',
    pad_char: "_",            // Ejemplo de caracter de relleno
    img_height: 70,           // Ejemplo: Altura de entrada del modelo
    img_width: 140            // Ejemplo: Ancho de entrada del modelo
    // --- FIN VALORES DE EJEMPLO ---
};

// --- Alternativa para Múltiples Modelos ---
/*
Si llegaras a tener varios modelos OCR (definidos en hub-ocr.js)
y cada uno necesitara una configuración diferente, podrías exportar
un objeto que mapee el nombre del modelo a su configuración:

export const OCR_CONFIGS = {
    "ocr-latin-v1": { // Nombre debe coincidir con el de AVAILABLE_OCR_MODELS
        max_plate_slots: 9,
        alphabet: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        pad_char: "-",
        img_height: 64,
        img_width: 128
    },
    "otro-modelo-ocr": {
        max_plate_slots: 7,
        alphabet: "0123456789",
        pad_char: "*",
        img_height: 32,
        img_width: 100
    }
    // ... otros modelos
};

*/
