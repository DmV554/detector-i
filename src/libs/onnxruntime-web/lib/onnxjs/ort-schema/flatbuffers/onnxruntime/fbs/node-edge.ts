// automatically generated by the FlatBuffers compiler, do not modify

/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */

import * as flatbuffers from 'flatbuffers';

import { EdgeEnd } from './edge-end.js';

export class NodeEdge {
  bb: flatbuffers.ByteBuffer | null = null;
  bb_pos = 0;
  __init(i: number, bb: flatbuffers.ByteBuffer): NodeEdge {
    this.bb_pos = i;
    this.bb = bb;
    return this;
  }

  static getRootAsNodeEdge(bb: flatbuffers.ByteBuffer, obj?: NodeEdge): NodeEdge {
    return (obj || new NodeEdge()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  static getSizePrefixedRootAsNodeEdge(bb: flatbuffers.ByteBuffer, obj?: NodeEdge): NodeEdge {
    bb.setPosition(bb.position() + flatbuffers.SIZE_PREFIX_LENGTH);
    return (obj || new NodeEdge()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  nodeIndex(): number {
    const offset = this.bb!.__offset(this.bb_pos, 4);
    return offset ? this.bb!.readUint32(this.bb_pos + offset) : 0;
  }

  inputEdges(index: number, obj?: EdgeEnd): EdgeEnd | null {
    const offset = this.bb!.__offset(this.bb_pos, 6);
    return offset
      ? (obj || new EdgeEnd()).__init(this.bb!.__vector(this.bb_pos + offset) + index * 12, this.bb!)
      : null;
  }

  inputEdgesLength(): number {
    const offset = this.bb!.__offset(this.bb_pos, 6);
    return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
  }

  outputEdges(index: number, obj?: EdgeEnd): EdgeEnd | null {
    const offset = this.bb!.__offset(this.bb_pos, 8);
    return offset
      ? (obj || new EdgeEnd()).__init(this.bb!.__vector(this.bb_pos + offset) + index * 12, this.bb!)
      : null;
  }

  outputEdgesLength(): number {
    const offset = this.bb!.__offset(this.bb_pos, 8);
    return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
  }

  static startNodeEdge(builder: flatbuffers.Builder) {
    builder.startObject(3);
  }

  static addNodeIndex(builder: flatbuffers.Builder, nodeIndex: number) {
    builder.addFieldInt32(0, nodeIndex, 0);
  }

  static addInputEdges(builder: flatbuffers.Builder, inputEdgesOffset: flatbuffers.Offset) {
    builder.addFieldOffset(1, inputEdgesOffset, 0);
  }

  static startInputEdgesVector(builder: flatbuffers.Builder, numElems: number) {
    builder.startVector(12, numElems, 4);
  }

  static addOutputEdges(builder: flatbuffers.Builder, outputEdgesOffset: flatbuffers.Offset) {
    builder.addFieldOffset(2, outputEdgesOffset, 0);
  }

  static startOutputEdgesVector(builder: flatbuffers.Builder, numElems: number) {
    builder.startVector(12, numElems, 4);
  }

  static endNodeEdge(builder: flatbuffers.Builder): flatbuffers.Offset {
    const offset = builder.endObject();
    return offset;
  }

  static createNodeEdge(
    builder: flatbuffers.Builder,
    nodeIndex: number,
    inputEdgesOffset: flatbuffers.Offset,
    outputEdgesOffset: flatbuffers.Offset,
  ): flatbuffers.Offset {
    NodeEdge.startNodeEdge(builder);
    NodeEdge.addNodeIndex(builder, nodeIndex);
    NodeEdge.addInputEdges(builder, inputEdgesOffset);
    NodeEdge.addOutputEdges(builder, outputEdgesOffset);
    return NodeEdge.endNodeEdge(builder);
  }
}
