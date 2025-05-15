/**
 * @fileoverview Postprocessing functions for detection models.
 * Equivalent to open_image_models/detection/core/yolo_v9/postprocess.py
 * MODIFIED to handle:
 * 1. Both 3D [batch, num_detections, details] and 2D [batch, details] tensor shapes.
 * 2. Batch sizes > 1, processing all items in the batch.
 */

// Import base classes assuming they are in the parent directory 'core'
// Adjust the path '../core/base.js' based on your actual file structure.
import { BoundingBox, DetectionResult } from '../base.js';

/**
 * Converts raw model prediction tensor data into a list of DetectionResult objects
 * for ALL items in the input batch, adjusting coordinates for padding and scaling
 * applied during preprocessing.
 *
 * Handles both 3D prediction shapes (e.g., [B, N, M]) for multiple detections per item,
 * and 2D prediction shapes (e.g., [B, M]) for a single detection per item.
 *
 * Assumes the prediction data layout per detection is:
 * [ignored, x1, y1, x2, y2, class_id, score, ...] (indices 1-6 used)
 * where coordinates are relative to the padded/scaled input image.
 *
 * Assumes the same `ratio` and `padding` were used for all items in the batch.
 *
 * @export
 * @param {Float32Array | number[]} predictionData - Flat array containing the prediction data from the model output tensor FOR ALL BATCH ITEMS.
 * @param {number[]} predictionShape - Shape of the prediction tensor (e.g., [batchSize, numDetections, dataPoints] or [batchSize, dataPoints]).
 * @param {string[]} classLabels - List of class labels corresponding to the class IDs.
 * @param {{w: number, h: number}} ratio - Scaling ratio {w, h} used during preprocessing (assumed same for all batch items).
 * @param {{w: number, h: number}} padding - Padding {w, h} added during preprocessing (assumed same for all batch items).
 * @param {number} [scoreThreshold=0.5] - Minimum confidence score to include a detection result.
 * @returns {DetectionResult[]} A single flat list containing DetectionResult objects for all valid detections across ALL items in the batch. The order corresponds to batch item 0, then batch item 1, etc.
 * @throws {Error} If predictionShape is invalid or inconsistent with data.
 */
export function convertToDetectionResult(
    predictionData,
    predictionShape,
    classLabels,
    ratio,
    padding,
    scoreThreshold = 0.5
) {
    // Stores results from ALL batch items flattened into one list
    const allResults = [];

    if (!Array.isArray(predictionShape) || predictionShape.length < 2 || predictionShape.length > 3) {
        throw new Error(`Invalid predictionShape: Expected 2 or 3 dimensions, got ${predictionShape?.length} (${predictionShape})`);
    }

    const batchSize = predictionShape[0];
    let numDetectionsPerItem, numDataPoints;
    let expectedDataLength;
    let dataPointsPerBatchItem; // Total data points for one item in the batch

    // --- Determine structure based on dimensions ---
    if (predictionShape.length === 3) {
        // Case 1: 3D Input -> [batchSize, numDetectionsPerItem, numDataPoints]
        numDetectionsPerItem = predictionShape[1];
        numDataPoints = predictionShape[2];
        dataPointsPerBatchItem = numDetectionsPerItem * numDataPoints;
        console.log(`Processing 3D input: Batch=${batchSize}, Detections/Item=${numDetectionsPerItem}, DataPoints/Detection=${numDataPoints}`);

        if (numDataPoints < 7) {
            throw new Error(`Insufficient data points per detection (${numDataPoints}) in 3D input. Need at least 7.`);
        }

    } else { // predictionShape.length === 2
        // Case 2: 2D Input -> [batchSize, numDataPoints] (Implies 1 detection per batch item)
        numDetectionsPerItem = 1; // Only one detection represented by each row
        numDataPoints = predictionShape[1];
        dataPointsPerBatchItem = numDataPoints; // Only M data points per batch item
        console.log(`Processing 2D input: Batch=${batchSize}, Detections/Item=1 (implied), DataPoints/Detection=${numDataPoints}`);

        if (numDataPoints < 7) {
            throw new Error(`Insufficient data points per detection (${numDataPoints}) in 2D input. Need at least 7.`);
        }
    }

    // --- Validate Total Data Length ---
    expectedDataLength = batchSize * dataPointsPerBatchItem;
    if (predictionData.length !== expectedDataLength) {
         throw new Error(`Prediction data length (${predictionData.length}) does not match expected length based on shape (${expectedDataLength}). Shape: [${predictionShape.join(', ')}]`);
    }


    // --- Process Detections for EACH item in the batch ---
    for (let b = 0; b < batchSize; b++) {
        // Calculate the starting index in the flat predictionData array for the current batch item 'b'
        const batchOffset = b * dataPointsPerBatchItem;

        // --- Inner loop for detections within the current batch item 'b' ---
        // Note: For 2D shape, numDetectionsPerItem is 1, so this loop runs once per batch item.
        for (let i = 0; i < numDetectionsPerItem; i++) {
            // Calculate the offset for the current detection 'i' RELATIVE to the start of the current batch item 'b's data
            const detectionOffsetWithinBatch = i * numDataPoints;
            // Calculate the FINAL offset in the flat predictionData array
            const finalOffset = batchOffset + detectionOffsetWithinBatch;

            // Extract score (index 6 relative to final offset)
            const score = predictionData[finalOffset + 6];

            // Filter by confidence threshold
            if (score < scoreThreshold) {
                continue; // Skip this detection
            }

            // Extract raw bounding box coordinates (indices 1 to 4 relative to final offset)
            const x1_raw = predictionData[finalOffset + 1];
            const y1_raw = predictionData[finalOffset + 2];
            const x2_raw = predictionData[finalOffset + 3];
            const y2_raw = predictionData[finalOffset + 4];

            // --- Coordinate Transformation ---
            // Assumes ratio and padding are the same for all items in the batch
            const x1 = (x1_raw - padding.w) / ratio.w;
            const y1 = (y1_raw - padding.h) / ratio.h;
            const x2 = (x2_raw - padding.w) / ratio.w;
            const y2 = (y2_raw - padding.h) / ratio.h;

             // Loguear solo para la primera detección del primer batch para empezar
            if (b === 0 && i === 0) {
                console.log(`DEBUG C2DR[b=${b}, i=${i}]: Raw Coords (x1r,y1r,x2r,y2r): ${x1_raw?.toFixed(2)}, ${y1_raw?.toFixed(2)}, ${x2_raw?.toFixed(2)}, ${y2_raw?.toFixed(2)}`);
                // Usamos ?.toFixed() por si acaso ratio/padding no tuvieran las props esperadas
                console.log(`DEBUG C2DR[b=${b}, i=${i}]: Usando ratio: w=${ratio?.w?.toFixed(4)}, h=${ratio?.h?.toFixed(4)}, padding: w=${padding?.w?.toFixed(2)}, h=${padding?.h?.toFixed(2)}`);
                console.log(`DEBUG C2DR[b=${b}, i=${i}]: Coords Calculadas (x1,y1,x2,y2): ${x1?.toFixed(2)}, ${y1?.toFixed(2)}, ${x2?.toFixed(2)}, ${y2?.toFixed(2)}`);
            }

            // Extract class ID (index 5 relative to final offset) - round to handle potential float values
            const classId = Math.round(predictionData[finalOffset + 5]);

            // Map class ID to label
            const label = (classId >= 0 && classId < classLabels.length)
                            ? classLabels[classId]
                            : `class_${classId}`; // Fallback

            // Create BoundingBox instance
            const boundingBox = new BoundingBox(x1, y1, x2, y2);

            // Create DetectionResult instance
            const detectionResult = new DetectionResult(label, score, boundingBox);

            // Add to the single flat list of results
            allResults.push(detectionResult);
        } // End inner loop (detections within batch item)
    } // End outer loop (batch items)

    // Return the flat list containing results from all batch items
    return allResults;
}

// --- Example Usage Notes ---
/*
// Now you can process outputs with batchSize > 1:

// Example: 3D Output with Batch Size 2
// - outputTensorDataMultiBatch3D: Float32Array representing results for 2 images
// - outputTensorShapeMultiBatch3D: [2, 100, 7] // Batch size 2

// Example: 2D Output with Batch Size 4 (e.g., 4 separate single detections)
// - outputTensorDataMultiBatch2D: Float32Array representing 4 results ([det1_data, det2_data, det3_data, det4_data])
// - outputTensorShapeMultiBatch2D: [4, 7] // Batch size 4

// Common parameters (assuming same for all batch items)
// - labels: ['license-plate', 'car']
// - preprocessingRatio: { w: 0.5, h: 0.5 }
// - preprocessingPadding: { w: 10, h: 10 }

const all_detections = convertToDetectionResult(
    outputTensorDataMultiBatch, // Your flattened data for multiple batches
    outputTensorShapeMultiBatch, // The shape [B, N, M] or [B, M]
    labels,
    preprocessingRatio,
    preprocessingPadding,
    0.4
);

// all_detections will be a flat list like:
// [ DetectionResult_Img0_Det0, DetectionResult_Img0_Det1, ..., DetectionResult_Img1_Det0, ... ]
console.log(`Total detections found across all batches: ${all_detections.length}`);
console.log(all_detections);
*/