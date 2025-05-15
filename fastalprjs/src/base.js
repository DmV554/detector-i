/**
 * @fileoverview Módulo base con estructuras de datos y clases abstractas.
 * Equivalente al módulo base.py de Python.
 */

/**
 * Representa un cuadro delimitador (Bounding Box) con coordenadas enteras.
 * Las instancias son inmutables después de su creación.
 * Reemplaza a la dataclass BoundingBox.
 */
export class BoundingBox {
    /** @type {number} Coordenada X de la esquina superior izquierda (entero). */
    x1;
    /** @type {number} Coordenada Y de la esquina superior izquierda (entero). */
    y1;
    /** @type {number} Coordenada X de la esquina inferior derecha (entero). */
    x2;
    /** @type {number} Coordenada Y de la esquina inferior derecha (entero). */
    y2;

    /**
     * Crea una instancia inmutable de BoundingBox.
     * Redondea las coordenadas a enteros.
     * @param {number} x1 - Coordenada X superior izquierda.
     * @param {number} y1 - Coordenada Y superior izquierda.
     * @param {number} x2 - Coordenada X inferior derecha.
     * @param {number} y2 - Coordenada Y inferior derecha.
     */
    constructor(x1, y1, x2, y2) {
        // Asegurar que sean enteros para consistencia con la definición Python
        this.x1 = Math.round(x1);
        this.y1 = Math.round(y1);
        this.x2 = Math.round(x2);
        this.y2 = Math.round(y2);
        // Hacer la instancia inmutable
        Object.freeze(this);
    }
}

/**
 * Representa el resultado de una única detección de objetos.
 * Las instancias son inmutables después de su creación.
 * Reemplaza a la dataclass DetectionResult.
 */
export class DetectionResult {
    /** @type {string} Etiqueta del objeto detectado (ej: 'License Plate'). */
    label;
    /** @type {number} Puntuación de confianza de la detección (normalmente entre 0 y 1). */
    confidence;
    /** @type {BoundingBox} El cuadro delimitador de la detección. */
    boundingBox; // Usamos camelCase en JS en lugar de snake_case

    /**
     * Crea una instancia inmutable de DetectionResult.
     * @param {string} label - Etiqueta del objeto.
     * @param {number} confidence - Confianza de la detección.
     * @param {BoundingBox} boundingBox - Instancia de la clase BoundingBox.
     * @throws {Error} Si boundingBox no es una instancia de BoundingBox.
     */
    constructor(label, confidence, boundingBox) {
        if (!(boundingBox instanceof BoundingBox)) {
            throw new Error("El argumento 'boundingBox' debe ser una instancia de la clase BoundingBox.");
        }
        this.label = label;
        this.confidence = confidence;
        this.boundingBox = boundingBox;
        // Hacer la instancia inmutable
        Object.freeze(this);
    }
}

/**
 * Representa el resultado de un proceso OCR.
 * Las instancias son inmutables después de su creación.
 * Reemplaza a la dataclass OcrResult.
 */
export class OcrResult {
     /** @type {string} El texto reconocido. */
    text;
    /** @type {number | Array<number>} Puntuación de confianza general O una lista de confianzas por caracter. */
    confidence;

    /**
     * Crea una instancia inmutable de OcrResult.
     * @param {string} text - El texto reconocido.
     * @param {number | Array<number>} confidence - Confianza general o por caracter.
     */
    constructor(text, confidence) {
        this.text = text;
        this.confidence = confidence;
        // Hacer la instancia inmutable
        Object.freeze(this);
    }
}


// --- Clases Base Abstractas (Simuladas) ---

/**
 * Clase base abstracta (simulada) para detectores de objetos.
 * Las subclases DEBEN implementar el método `predict`.
 * También pueden implementar `initialize` si necesitan carga asíncrona.
 * @abstract
 */
export class BaseDetector {
    /**
     * El constructor previene la instanciación directa de la clase base.
     */
    constructor() {
        if (this.constructor === BaseDetector) {
            throw new Error("La clase abstracta 'BaseDetector' no puede ser instanciada directamente.");
        }
        // Podrías inicializar propiedades comunes aquí si las hubiera
    }

    /**
     * Realiza la detección en la fuente de imagen de entrada.
     * Este método DEBE ser sobrescrito por las subclases.
     * @abstract
     * @param {ImageData | HTMLImageElement | HTMLCanvasElement | Blob | File | cv.Mat} imageSource - La imagen de entrada. Se aceptan varios formatos comunes del navegador o un cv.Mat si se usa OpenCV.
     * @returns {Promise<Array<DetectionResult>>} Una promesa que se resuelve con una lista de objetos DetectionResult.
     * @throws {Error} Si el método no es implementado por la subclase.
     */
    async predict(imageSource) {
        // Nota: imageSource es solo un parámetro de ejemplo, puede variar.
        throw new Error("El método 'predict(imageSource)' debe ser implementado por la subclase.");
    }

    /**
     * Método opcional para inicializar el modelo de forma asíncrona (ej. cargar pesos).
     * Puede ser sobrescrito por las subclases si es necesario.
     * @abstract
     * @returns {Promise<void>}
     */
    async initialize() {
        // La implementación por defecto no hace nada.
        // Alternativamente, podrías lanzar un error si la inicialización es obligatoria:
        // throw new Error("El método 'initialize()' debe ser implementado por la subclase.");
        console.warn(`${this.constructor.name}.initialize() no implementado (opcional).`);
    }
}


/**
 * Clase base abstracta (simulada) para implementaciones de OCR.
 * Las subclases DEBEN implementar el método `predict`.
 * También pueden implementar `initialize` si necesitan carga asíncrona.
 * @abstract
 */
export class BaseOcr {
     /**
     * El constructor previene la instanciación directa de la clase base.
     */
     constructor() {
        if (this.constructor === BaseOcr) {
            throw new Error("La clase abstracta 'BaseOcr' no puede ser instanciada directamente.");
        }
        // Inicializar propiedades comunes si las hubiera
    }

    /**
     * Realiza OCR en una imagen recortada de una matrícula.
     * Este método DEBE ser sobrescrito por las subclases.
     * @abstract
     * @param {ImageData | HTMLImageElement | HTMLCanvasElement | Blob | File | cv.Mat} croppedPlateSource - La imagen recortada de la matrícula.
     * @returns {Promise<OcrResult | null>} Una promesa que se resuelve con un objeto OcrResult o null si el reconocimiento falla o no es aplicable.
     * @throws {Error} Si el método no es implementado por la subclase.
     */
    async predict(croppedPlateSource) {
         // Nota: croppedPlateSource es solo un parámetro de ejemplo.
        throw new Error("El método 'predict(croppedPlateSource)' debe ser implementado por la subclase.");
    }

    /**
     * Método opcional para inicializar el modelo de forma asíncrona (ej. cargar pesos).
     * Puede ser sobrescrito por las subclases si es necesario.
     * @abstract
     * @returns {Promise<void>}
     */
    async initialize() {
        console.warn(`${this.constructor.name}.initialize() no implementado (opcional).`);
    }
}