/**
 * @fileoverview Implementación por defecto del detector, usando LicensePlateDetector.
 */

// Importar clases/interfaces base de nuestra nueva librería
import { BaseDetector, BoundingBox, DetectionResult } from './base.js';

// Importar la implementación CONCRETA del detector que vamos a envolver
// ¡¡ASEGÚRATE QUE ESTA RUTA SEA CORRECTA!! Es la ruta a tu detector anterior.
import { LicensePlateDetector } from '/detector-i/open-image-models-js/pipeline/license_plate.js';

// Opcional: importar tipos para documentación, si los tienes definidos
// import { PlateDetectorModelName } from './hub.js'; // Del hub del DETECTOR
// import * as ort from 'onnxruntime-web'; // Para ort.SessionOptions

/**
 * Implementación por defecto de BaseDetector que utiliza la clase
 * LicensePlateDetector existente (de open-image-models-js) internamente.
 *
 * @export
 * @class DefaultDetector
 * @extends {BaseDetector}
 */
export class DefaultDetector extends BaseDetector {
    /**
     * Instancia interna del detector envuelto.
     * @type {LicensePlateDetector | null}
     * @private // Indicativo de que es para uso interno
     */
    _detectorInstance = null; // La renombramos para evitar confusión con BaseDetector

    /** @type {string} Nombre del modelo a cargar. */
    modelName;
    /** @type {number} Umbral de confianza. */
    confThresh;
    /** @type {string[] | undefined} Proveedores para ONNX Runtime. */
    providers;
     /** @type {ort.SessionOptions | undefined} Opciones de sesión para ONNX Runtime. */
    sessionOptions;

    /**
     * Configura el DefaultDetector. La inicialización real (carga del modelo)
     * ocurre de forma asíncrona al llamar a `initialize()`.
     *
     * @param {object} [options={}] - Opciones de configuración.
     * @param {string} [options.modelName='yolo-v9-t-384-license-plate-end2end'] - Nombre del modelo detector (debe existir en el hub del detector).
     * @param {number} [options.confThresh=0.4] - Umbral de confianza para la detección.
     * @param {string[]} [options.providers] - Proveedores de ejecución ONNX Runtime opcionales (ej. ['webgl', 'wasm']). Si no se especifica, usa los defaults de LicensePlateDetector.
     * @param {any} [options.sessionOptions] - Opciones de sesión ONNX Runtime opcionales. Si no se especifica, usa los defaults de LicensePlateDetector.
     */
    constructor(options = {}) {
        super(); // Llama al constructor de BaseDetector (que verifica que no se instancie BaseDetector)
        this.modelName = options.modelName || 'yolo-v9-t-384-license-plate-end2end'; // Default como en Python
        // Usar ?? (nullish coalescing) por si 0 fuera un umbral válido
        this.confThresh = options.confThresh ?? 0.4;
        this.providers = options.providers; // Se pasa undefined si no se provee
        this.sessionOptions = options.sessionOptions; // Se pasa undefined si no se provee
        this._detectorInstance = null; // Marcar como no inicializado

        console.log(`DefaultDetector configurado: Modelo=${this.modelName}, Umbral=${this.confThresh}`);
    }

    /**
     * Inicializa el modelo detector subyacente (`LicensePlateDetector`) de forma asíncrona.
     * Debe llamarse antes de usar `predict`.
     * @override // Indica que estamos sobrescribiendo un método de BaseDetector
     * @returns {Promise<void>}
     * @throws {Error} Si la carga del modelo o la creación de la sesión falla.
     */
    async initialize() {
        if (this._detectorInstance) {
            console.warn("El detector subyacente ya está inicializado.");
            return;
        }
        console.log(`Inicializando el LicensePlateDetector subyacente (${this.modelName})...`);
        try {
             // Preparamos las opciones para el método `create` de la clase que envolvemos
             const wrappedDetectorOptions = {
                  confThresh: this.confThresh,
                  // Pasamos providers y sessionOptions (pueden ser undefined, LicensePlateDetector usará sus defaults)
                  providers: this.providers,
                  sessionOptions: this.sessionOptions
             };
             // Llamamos al método estático `create` de la clase LicensePlateDetector importada
            this._detectorInstance = await LicensePlateDetector.create(this.modelName, wrappedDetectorOptions);
            console.log("LicensePlateDetector subyacente inicializado correctamente.");
        } catch (error) {
            console.error(`Fallo al inicializar LicensePlateDetector (${this.modelName}):`, error);
            this._detectorInstance = null; // Asegurar que esté null si falla
            throw error; // Relanzar el error para que el llamador se entere
        }
    }

    /**
     * Realiza la detección en la imagen de entrada usando la instancia envuelta
     * de `LicensePlateDetector` y transforma los resultados al formato de `base.js`.
     * @override // Indica que estamos sobrescribiendo un método de BaseDetector
     * @param {ImageData | HTMLImageElement | HTMLCanvasElement | Blob | File | cv.Mat} imageSource - La imagen de entrada.
     * @returns {Promise<Array<DetectionResult>>} Una promesa que se resuelve con una lista de objetos DetectionResult (definidos en base.js).
     * @throws {Error} Si el detector no está inicializado o si la predicción falla.
     */
    async predict(imageSource) {
        if (!this._detectorInstance) {
            throw new Error("El DefaultDetector no ha sido inicializado. Llama a initialize() primero.");
        }

        console.debug("Llamando a predict() del LicensePlateDetector envuelto...");
        // Llama al método predict de la instancia envuelta
        const underlyingDetections = await this._detectorInstance.predict(imageSource);
        console.debug(`Se recibieron ${underlyingDetections.length} detecciones del detector envuelto.`);

        // --- Transformación Crucial ---
        // Mapear los resultados devueltos por `_detectorInstance.predict`
        // a instancias de las clases `DetectionResult` y `BoundingBox` de nuestro `base.js`.
        // Asumimos que `underlyingDetections` es un array de objetos con propiedades
        // `label`, `confidence`, y `boundingBox` (que a su vez tiene `x1`, `y1`, `x2`, `y2`).
        const results = underlyingDetections.map(det => {
            // Añadir validación por si el detector envuelto devuelve algo inesperado
            if (!det || typeof det !== 'object' || !det.boundingBox || typeof det.boundingBox !== 'object') {
                console.warn("Se recibió un objeto de detección inválido del detector envuelto:", det);
                return null; // Marcar para filtrar después
            }
            try {
                 const bb = new BoundingBox(
                    det.boundingBox.x1,
                    det.boundingBox.y1,
                    det.boundingBox.x2,
                    det.boundingBox.y2
                 );
                 // Usar ?? para defaults por si label/confidence son null/undefined
                 return new DetectionResult(
                    det.label ?? 'unknown',
                    det.confidence ?? 0.0,
                    bb
                 );
            } catch (mapError) {
                 console.error("Error al mapear la detección envuelta:", det, mapError);
                 return null; // Marcar para filtrar si hay error en la creación
            }

        }).filter(res => res !== null); // Eliminar cualquier resultado nulo/inválido

        console.debug(`Transformadas ${results.length} detecciones al formato de la librería.`);
        return results;
    }
}