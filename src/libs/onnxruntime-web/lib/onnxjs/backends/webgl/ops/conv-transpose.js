'use strict';
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, '__esModule', { value: true });
exports.parseConvTransposeAttributes = exports.convTranspose = void 0;
const attribute_with_cache_key_1 = require('../../../attribute-with-cache-key');
const glsl_source_1 = require('../glsl-source');
const types_1 = require('../types');
const fuse_utils_1 = require('./fuse-utils');
const computeTotalPad = (inDim, stride, adj, kernel, dilation, outSize) =>
  (inDim - 1) * stride + adj + (kernel - 1) * dilation + 1 - outSize;
const distributePadding = (totalPad, autoPad, pads, head, tail) => {
  const smallPad = Math.floor(totalPad / 2);
  if (autoPad === 'SAME_UPPER') {
    pads[head] = smallPad;
    pads[tail] = totalPad - smallPad;
  } else if (autoPad === 'SAME_LOWER') {
    pads[head] = totalPad - smallPad;
    pads[tail] = smallPad;
  }
};
const calculateOutputShapeAndPads = (
  inputShape,
  kernelShape,
  dilations,
  autoPad,
  pads,
  strides,
  outputPadding,
  outputShape,
) => {
  const spatialRank = inputShape.length - 2;
  const updateShape = outputShape.length === 0;
  for (let i = 0; i < spatialRank; ++i) {
    const outSize = updateShape ? inputShape[i + 2] * strides[i] : outputShape[i];
    const totalPad = computeTotalPad(inputShape[i + 2], strides[i], pads[i], kernelShape[i], dilations[i], outSize);
    distributePadding(totalPad, autoPad, pads, i, i + spatialRank);
    if (updateShape) {
      outputShape.push(
        strides[i] * (inputShape[i + 2] - 1) +
          outputPadding[i] +
          (kernelShape[i] - 1) * dilations[i] +
          1 -
          pads[i] -
          pads[i + spatialRank],
      );
    }
  }
};
const convTranspose = (inferenceHandler, inputs, attributes) => {
  validateInputs(inputs, attributes); // currently will fail if not convTranspose2D
  return convTranspose2d(inferenceHandler, inputs, attributes);
};
exports.convTranspose = convTranspose;
const convTranspose2d = (inferenceHandler, inputs, attributes) => {
  const adjustedAttributes = getAdjustedConvTransposeAttributes(attributes, inputs);
  return [convTranspose2DUnpacked(inferenceHandler, inputs, adjustedAttributes)];
};
const createConvTransposeProgramMetadata = (hasBias, cacheHint) => ({
  name: 'ConvTranspose',
  inputNames: hasBias ? ['X', 'W', 'B'] : ['X', 'W'],
  inputTypes: hasBias
    ? [types_1.TextureType.unpacked, types_1.TextureType.unpacked, types_1.TextureType.unpacked]
    : [types_1.TextureType.unpacked, types_1.TextureType.unpacked],
  cacheHint,
});
const createUnpackedConvTransposeProgramInfo = (inferenceHandler, inputs, metadata, attributes) => {
  const hasBias = inputs.length > 2;
  const valueInit = hasBias ? 'getB(output_channel)' : '0.0';
  const xShape = inputs[0].dims;
  const wShape = inputs[1].dims;
  const outputChannelsPerGroup = wShape[1];
  const inputChannelsPerGroup = wShape[0] / attributes.group;
  const outputShape = [inputs[0].dims[0], inputs[1].dims[1] * attributes.group, ...attributes.outputShape];
  const glsl = (0, glsl_source_1.getGlsl)(inferenceHandler.session.backend.glContext.version);
  const { activationFunction, applyActivation } = (0, fuse_utils_1.getActivationSnippet)(attributes);
  const shaderSource = `
  const ivec2 strides = ivec2(${attributes.strides[0]}, ${attributes.strides[1]});
  const ivec2 pads = ivec2(${attributes.pads[0]}, ${attributes.pads[1]});
  ${activationFunction}
  void main() {
    ivec4 coords = getOutputCoords();
    int batch = coords.x;
    int output_channel = coords.y;

    ivec2 loc = coords.zw + pads;

    int group_id = output_channel / ${outputChannelsPerGroup};
    int wOutChannel = output_channel - group_id * ${outputChannelsPerGroup};

    float value = ${valueInit};
    for (int inChannelOffset = 0; inChannelOffset < ${inputChannelsPerGroup}; inChannelOffset++) {
      int input_channel = group_id * ${inputChannelsPerGroup} + inChannelOffset;
      for (int wWOff = 0; wWOff < ${wShape[2]}; wWOff++) {
        for (int wHOff = 0; wHOff < ${wShape[3]}; wHOff++) {
          ivec2 wOff = ivec2(wWOff * ${attributes.dilations[0]}, wHOff * ${attributes.dilations[1]});
          ivec2 wLoc = loc - wOff;
          ivec2 wLocIn = wLoc / strides;
          if (
            wLocIn * strides == wLoc &&
            wLocIn.x >= 0 && wLocIn.x < ${xShape[2]} &&
            wLocIn.y >= 0 && wLocIn.y < ${xShape[3]}
          ) {
            float xVal = getX(batch, input_channel, wLocIn.y, wLocIn.x);
            float wVal = getW(input_channel, wOutChannel, wHOff, wWOff);
            value += xVal * wVal;
          }
        }
      }
    }
    ${applyActivation}
    ${glsl.output} = vec4(value, .0, .0, .0);
  }
`;
  return {
    ...metadata,
    output: { dims: outputShape, type: inputs[0].type, textureType: types_1.TextureType.unpacked },
    shaderSource,
    hasMain: true,
  };
};
const createUnpackedConvTransposeProgramInfoLoader = (inferenceHandler, inputs, attributes) => {
  const metadata = createConvTransposeProgramMetadata(inputs.length > 2, attributes.cacheKey);
  return {
    ...metadata,
    get: () => createUnpackedConvTransposeProgramInfo(inferenceHandler, inputs, metadata, attributes),
  };
};
const convTranspose2DUnpacked = (inferenceHandler, inputs, attributes) => {
  const result = inferenceHandler.run(
    createUnpackedConvTransposeProgramInfoLoader(inferenceHandler, inputs, attributes),
    inputs,
  );
  return result;
};
const getAdjustedConvTransposeAttributes = (attributes, inputs) => {
  const kernelShape = attributes.kernelShape.slice();
  // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
  if (attributes.kernelShape.length === 0) {
    for (let i = 2; i < inputs[1].dims.length; ++i) {
      kernelShape.push(inputs[1].dims[i]);
    }
  }
  const pads = attributes.pads.slice();
  const outputShape = attributes.outputShape.slice();
  const inputShape = inputs[0].dims;
  // If outputShape is not specified in the attributes of this op, infer it from the parameters
  // Similarly, automatically infer pads if not specified
  calculateOutputShapeAndPads(
    inputShape,
    kernelShape,
    attributes.dilations,
    attributes.autoPad,
    pads,
    attributes.strides,
    attributes.outputPadding,
    outputShape,
  );
  // always return a new object so does not modify the original attributes
  const newAttributes = Object.assign({}, attributes);
  Object.assign(newAttributes, { kernelShape, pads, outputShape, cacheKey: attributes.cacheKey });
  return newAttributes;
};
const parseConvTransposeAttributes = (node) => {
  const attributes = node.attributes;
  const activationAttributes = (0, fuse_utils_1.parseInternalActivationAttributes)(attributes);
  // TODO : Make this generic enough to compute default attributes for multi-dimensional conv
  const autoPad = attributes.getString('auto_pad', 'NOTSET');
  const dilations = attributes.getInts('dilations', [1, 1]);
  const group = attributes.getInt('group', 1);
  const kernelShape = attributes.getInts('kernel_shape', []);
  const outputPadding = attributes.getInts('output_padding', [0, 0]);
  const outputShape = attributes.getInts('output_shape', []);
  const pads = attributes.getInts('pads', [0, 0, 0, 0]);
  const strides = attributes.getInts('strides', [1, 1]);
  return (0, attribute_with_cache_key_1.createAttributeWithCacheKey)({
    autoPad,
    dilations,
    group,
    kernelShape,
    outputPadding,
    outputShape,
    pads,
    strides,
    ...activationAttributes,
  });
};
exports.parseConvTransposeAttributes = parseConvTransposeAttributes;
const validateInputs = (inputs, attributes) => {
  // Refer to the below link for all input checks
  // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
  if (!inputs || (inputs.length !== 2 && inputs.length !== 3)) {
    throw new Error('Conv requires 2 or 3 inputs');
  }
  // TODO : Need to add support for multi-dimensional conv
  if (inputs[0].dims.length !== 4 || inputs[1].dims.length !== 4) {
    throw new Error('currently only support 2-dimensional conv');
  }
  // FILTER_IN_CHANNEL should be equal to DATA_CHANNEL
  const dataChannel = inputs[0].dims[1];
  const filterInChannel = inputs[1].dims[0];
  if (dataChannel !== filterInChannel) {
    throw new Error('FILTER_IN_CHANNEL should be equal to DATA_CHANNEL');
  }
  const featureMaps = inputs[1].dims[1] * attributes.group;
  // if bias is provided it should be 1D and the number of elements should be equal to the number of feature maps
  if (inputs.length === 3 && (inputs[2].dims.length !== 1 || inputs[2].dims[0] !== featureMaps)) {
    throw new Error('invalid bias');
  }
  const spatialRank = inputs[0].dims.length - 2;
  // wrong dilations dimension
  if (attributes.dilations.length !== spatialRank) {
    throw new Error(`dilations should be ${spatialRank}D`);
  }
  // Wrong strides dimension
  if (attributes.strides.length !== spatialRank) {
    throw new Error(`strides should be ${spatialRank}D`);
  }
  // Wrong pads dimension
  if (attributes.pads.length !== spatialRank * 2) {
    throw new Error(`pads should be ${spatialRank * 2}D`);
  }
  // Wrong output padding dimension
  if (attributes.outputPadding.length !== spatialRank) {
    throw new Error(`output_padding should be ${spatialRank}D`);
  }
  // if kernelShape is specified, it's data length must be 2 less than dims length of the weights tensor
  // (the first 2 dims are batch_size and channels)
  if (attributes.kernelShape.length !== 0 && attributes.kernelShape.length !== inputs[1].dims.length - 2) {
    throw new Error('invalid kernel shape');
  }
  // as with kernelShape, must have same number of spatial dims as input
  if (attributes.outputShape.length !== 0 && attributes.outputShape.length !== inputs[0].dims.length - 2) {
    throw new Error('invalid output shape');
  }
  // TODO : Need to add support for float64
  if (inputs[0].type !== 'float32' || inputs[1].type !== 'float32') {
    throw new Error('ConvTranspose input(X,W) should be float tensor');
  }
  if (inputs.length === 3 && inputs[2].type !== 'float32') {
    throw new Error('ConvTranspose input(bias) should be float tensor');
  }
};
//# sourceMappingURL=conv-transpose.js.map
