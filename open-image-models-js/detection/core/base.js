/**
 * @fileoverview Base classes and data structures for detection models.
 * Equivalent to open_image_models/detection/core/base.py
 */

/**
 * Represents a bounding box with top-left and bottom-right coordinates.
 * Attempts to be immutable after creation using Object.freeze().
 */
export class BoundingBox {
    /**
     * X-coordinate of the top-left corner.
     * @type {number}
     * @readonly
     */
    x1;
    /**
     * Y-coordinate of the top-left corner.
     * @type {number}
     * @readonly
     */
    y1;
    /**
     * X-coordinate of the bottom-right corner.
     * @type {number}
     * @readonly
     */
    x2;
    /**
     * Y-coordinate of the bottom-right corner.
     * @type {number}
     * @readonly
     */
    y2;

    /**
     * Creates an instance of BoundingBox.
     * Coordinates are rounded to the nearest integer upon creation.
     * @param {number} x1 - X-coordinate of the top-left corner.
     * @param {number} y1 - Y-coordinate of the top-left corner.
     * @param {number} x2 - X-coordinate of the bottom-right corner.
     * @param {number} y2 - Y-coordinate of the bottom-right corner.
     */
    constructor(x1, y1, x2, y2) {
        this.x1 = Math.round(x1);
        this.y1 = Math.round(y1);
        this.x2 = Math.round(x2);
        this.y2 = Math.round(y2);
        Object.freeze(this); // Make instance immutable
    }

    /**
     * Returns the width of the bounding box.
     * @returns {number}
     */
    get width() {
        return this.x2 - this.x1;
    }

    /**
     * Returns the height of the bounding box.
     * @returns {number}
     */
    get height() {
        return this.y2 - this.y1;
    }

    /**
     * Returns the area of the bounding box.
     * @returns {number}
     */
    get area() {
        return this.width * this.height;
    }

    /**
     * Returns the (x, y) coordinates of the center of the bounding box.
     * @returns {{cx: number, cy: number}}
     */
    get center() {
        const cx = (this.x1 + this.x2) / 2.0;
        const cy = (this.y1 + this.y2) / 2.0;
        return { cx, cy };
    }

    /**
     * Returns the intersection of this bounding box with another bounding box.
     * If they do not intersect, returns null.
     * @param {BoundingBox} other - The other bounding box.
     * @returns {BoundingBox | null} The intersecting BoundingBox or null.
     */
    intersection(other) {
        const x1 = Math.max(this.x1, other.x1);
        const y1 = Math.max(this.y1, other.y1);
        const x2 = Math.min(this.x2, other.x2);
        const y2 = Math.min(this.y2, other.y2);

        if (x2 > x1 && y2 > y1) {
            // Coordinates will be integers if inputs were integers
            return new BoundingBox(x1, y1, x2, y2);
        }

        return null;
    }

    /**
     * Computes the Intersection-over-Union (IoU) between this bounding box
     * and another bounding box.
     * @param {BoundingBox} other - The other bounding box.
     * @returns {number} The IoU value (between 0.0 and 1.0).
     */
    iou(other) {
        const inter = this.intersection(other);

        if (inter === null) {
            return 0.0;
        }

        const interArea = inter.area;
        const unionArea = this.area + other.area - interArea;
        return unionArea > 0 ? interArea / unionArea : 0.0;
    }

    /**
     * Converts bounding box to [x, y, width, height] format,
     * where (x, y) is the top-left corner.
     * @returns {[number, number, number, number]} Array [x1, y1, width, height].
     */
    toXYWH() {
        return [this.x1, this.y1, this.width, this.height];
    }

    /**
     * Allows direct iteration over the bounding box coordinates [x1, y1, x2, y2].
     * Example: `for (const coord of boundingBox) { ... }` or `[x1, y1, x2, y2] = boundingBox;`
     * @returns {Generator<number, void, undefined>}
     */
    * [Symbol.iterator]() {
        yield this.x1;
        yield this.y1;
        yield this.x2;
        yield this.y2;
    }

    /**
     * Returns a new `BoundingBox` with coordinates clamped within the range
     * [0, max_width] and [0, max_height].
     * @param {number} maxWidth - The maximum width boundary.
     * @param {number} maxHeight - The maximum height boundary.
     * @returns {BoundingBox} A new, clamped `BoundingBox`.
     */
    clamp(maxWidth, maxHeight) {
        // Use Math.max/min for clamping, then round just in case maxWidth/maxHeight aren't integers
        const x1 = Math.round(Math.max(0, Math.min(this.x1, maxWidth)));
        const y1 = Math.round(Math.max(0, Math.min(this.y1, maxHeight)));
        const x2 = Math.round(Math.max(0, Math.min(this.x2, maxWidth)));
        const y2 = Math.round(Math.max(0, Math.min(this.y2, maxHeight)));
        return new BoundingBox(x1, y1, x2, y2);
    }

   /**
     * Checks if the bounding box is valid by ensuring that:
     * 1. The coordinates are in the correct order (x1 < x2 and y1 < y2).
     * 2. The bounding box lies entirely within the frame boundaries [0, frameWidth], [0, frameHeight].
     * Note: Python original check <= frameWidth/Height which means a box touching the edge is valid.
     * @param {number} frameWidth - The width of the frame.
     * @param {number} frameHeight - The height of the frame.
     * @returns {boolean} True if the bounding box is valid, False otherwise.
     */
    isValid(frameWidth, frameHeight) {
        return (
            this.x1 < this.x2 && this.y1 < this.y2 && // Check order first
            this.x1 >= 0 && this.y1 >= 0 && // Check lower bounds
            this.x2 <= frameWidth && this.y2 <= frameHeight // Check upper bounds
        );
    }
}


/**
 * Represents the result of an object detection.
 * Attempts to be immutable after creation using Object.freeze().
 */
export class DetectionResult {
    /**
     * Detected object label.
     * @type {string}
     * @readonly
     */
    label;
    /**
     * Confidence score of the detection (typically between 0.0 and 1.0).
     * @type {number}
     * @readonly
     */
    confidence;
    /**
     * Bounding box of the detected object.
     * @type {BoundingBox}
     * @readonly
     */
    boundingBox;

    /**
     * Creates an instance of DetectionResult.
     * @param {string} label - Detected object label.
     * @param {number} confidence - Confidence score of the detection.
     * @param {BoundingBox} boundingBox - Bounding box of the detected object.
     */
    constructor(label, confidence, boundingBox) {
        this.label = label;
        this.confidence = confidence;
        this.boundingBox = boundingBox; // Assumes boundingBox is already a BoundingBox instance
        Object.freeze(this); // Make instance immutable
    }

}


/**
 * Abstract base class (similar concept to Python Protocol) for Object Detectors.
 * Defines the interface that specific detector implementations should follow.
 * Methods should be overridden by subclasses.
 */
export class ObjectDetector {

    /**
     * Perform object detection on one or multiple images.
     * Subclasses must implement this method.
     *
     * @abstract
     * @param {any} images - Input image(s). Type depends on implementation
     * (e.g., HTMLImageElement, HTMLCanvasElement, ImageData, Tensor, path string, or arrays of these).
     * @returns {Promise<DetectionResult[] | DetectionResult[][]>} A promise resolving to:
     * - An array of DetectionResult for a single image input.
     * - An array of arrays of DetectionResult for multiple image inputs.
     * @throws {Error} If not implemented by subclass.
     */
    async predict(images) {
        throw new Error("Method 'predict()' must be implemented by subclasses.");
    }


    /**
     * Run object detection and display the predictions on the image.
     * Subclasses should implement this method for visualization.
     *
     * @abstract
     * @param {any} image - An input image (e.g., HTMLImageElement, HTMLCanvasElement, ImageData).
     * @returns {Promise<any>} A promise resolving to the image with detections drawn
     * (e.g., HTMLCanvasElement, ImageData). Type depends on implementation.
     * @throws {Error} If not implemented by subclass.
     */
    async displayPredictions(image) {
        throw new Error("Method 'displayPredictions()' must be implemented by subclasses.");
    }
}