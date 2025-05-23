'use strict';
// automatically generated by the FlatBuffers compiler, do not modify
var __createBinding =
  (this && this.__createBinding) ||
  (Object.create
    ? function (o, m, k, k2) {
        if (k2 === undefined) k2 = k;
        var desc = Object.getOwnPropertyDescriptor(m, k);
        if (!desc || ('get' in desc ? !m.__esModule : desc.writable || desc.configurable)) {
          desc = {
            enumerable: true,
            get: function () {
              return m[k];
            },
          };
        }
        Object.defineProperty(o, k2, desc);
      }
    : function (o, m, k, k2) {
        if (k2 === undefined) k2 = k;
        o[k2] = m[k];
      });
var __setModuleDefault =
  (this && this.__setModuleDefault) ||
  (Object.create
    ? function (o, v) {
        Object.defineProperty(o, 'default', { enumerable: true, value: v });
      }
    : function (o, v) {
        o['default'] = v;
      });
var __importStar =
  (this && this.__importStar) ||
  function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null)
      for (var k in mod)
        if (k !== 'default' && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
  };
Object.defineProperty(exports, '__esModule', { value: true });
exports.NodesToOptimizeIndices = void 0;
/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */
const flatbuffers = __importStar(require('flatbuffers'));
/**
 * nodes to consider for a runtime optimization
 * see corresponding type in onnxruntime/core/graph/runtime_optimization_record.h
 */
class NodesToOptimizeIndices {
  constructor() {
    this.bb = null;
    this.bb_pos = 0;
  }
  __init(i, bb) {
    this.bb_pos = i;
    this.bb = bb;
    return this;
  }
  static getRootAsNodesToOptimizeIndices(bb, obj) {
    return (obj || new NodesToOptimizeIndices()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }
  static getSizePrefixedRootAsNodesToOptimizeIndices(bb, obj) {
    bb.setPosition(bb.position() + flatbuffers.SIZE_PREFIX_LENGTH);
    return (obj || new NodesToOptimizeIndices()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }
  nodeIndices(index) {
    const offset = this.bb.__offset(this.bb_pos, 4);
    return offset ? this.bb.readUint32(this.bb.__vector(this.bb_pos + offset) + index * 4) : 0;
  }
  nodeIndicesLength() {
    const offset = this.bb.__offset(this.bb_pos, 4);
    return offset ? this.bb.__vector_len(this.bb_pos + offset) : 0;
  }
  nodeIndicesArray() {
    const offset = this.bb.__offset(this.bb_pos, 4);
    return offset
      ? new Uint32Array(
          this.bb.bytes().buffer,
          this.bb.bytes().byteOffset + this.bb.__vector(this.bb_pos + offset),
          this.bb.__vector_len(this.bb_pos + offset),
        )
      : null;
  }
  numInputs() {
    const offset = this.bb.__offset(this.bb_pos, 6);
    return offset ? this.bb.readUint32(this.bb_pos + offset) : 0;
  }
  numOutputs() {
    const offset = this.bb.__offset(this.bb_pos, 8);
    return offset ? this.bb.readUint32(this.bb_pos + offset) : 0;
  }
  hasVariadicInput() {
    const offset = this.bb.__offset(this.bb_pos, 10);
    return offset ? !!this.bb.readInt8(this.bb_pos + offset) : false;
  }
  hasVariadicOutput() {
    const offset = this.bb.__offset(this.bb_pos, 12);
    return offset ? !!this.bb.readInt8(this.bb_pos + offset) : false;
  }
  numVariadicInputs() {
    const offset = this.bb.__offset(this.bb_pos, 14);
    return offset ? this.bb.readUint32(this.bb_pos + offset) : 0;
  }
  numVariadicOutputs() {
    const offset = this.bb.__offset(this.bb_pos, 16);
    return offset ? this.bb.readUint32(this.bb_pos + offset) : 0;
  }
  static startNodesToOptimizeIndices(builder) {
    builder.startObject(7);
  }
  static addNodeIndices(builder, nodeIndicesOffset) {
    builder.addFieldOffset(0, nodeIndicesOffset, 0);
  }
  static createNodeIndicesVector(builder, data) {
    builder.startVector(4, data.length, 4);
    for (let i = data.length - 1; i >= 0; i--) {
      builder.addInt32(data[i]);
    }
    return builder.endVector();
  }
  static startNodeIndicesVector(builder, numElems) {
    builder.startVector(4, numElems, 4);
  }
  static addNumInputs(builder, numInputs) {
    builder.addFieldInt32(1, numInputs, 0);
  }
  static addNumOutputs(builder, numOutputs) {
    builder.addFieldInt32(2, numOutputs, 0);
  }
  static addHasVariadicInput(builder, hasVariadicInput) {
    builder.addFieldInt8(3, +hasVariadicInput, +false);
  }
  static addHasVariadicOutput(builder, hasVariadicOutput) {
    builder.addFieldInt8(4, +hasVariadicOutput, +false);
  }
  static addNumVariadicInputs(builder, numVariadicInputs) {
    builder.addFieldInt32(5, numVariadicInputs, 0);
  }
  static addNumVariadicOutputs(builder, numVariadicOutputs) {
    builder.addFieldInt32(6, numVariadicOutputs, 0);
  }
  static endNodesToOptimizeIndices(builder) {
    const offset = builder.endObject();
    return offset;
  }
  static createNodesToOptimizeIndices(
    builder,
    nodeIndicesOffset,
    numInputs,
    numOutputs,
    hasVariadicInput,
    hasVariadicOutput,
    numVariadicInputs,
    numVariadicOutputs,
  ) {
    NodesToOptimizeIndices.startNodesToOptimizeIndices(builder);
    NodesToOptimizeIndices.addNodeIndices(builder, nodeIndicesOffset);
    NodesToOptimizeIndices.addNumInputs(builder, numInputs);
    NodesToOptimizeIndices.addNumOutputs(builder, numOutputs);
    NodesToOptimizeIndices.addHasVariadicInput(builder, hasVariadicInput);
    NodesToOptimizeIndices.addHasVariadicOutput(builder, hasVariadicOutput);
    NodesToOptimizeIndices.addNumVariadicInputs(builder, numVariadicInputs);
    NodesToOptimizeIndices.addNumVariadicOutputs(builder, numVariadicOutputs);
    return NodesToOptimizeIndices.endNodesToOptimizeIndices(builder);
  }
}
exports.NodesToOptimizeIndices = NodesToOptimizeIndices;
//# sourceMappingURL=nodes-to-optimize-indices.js.map
