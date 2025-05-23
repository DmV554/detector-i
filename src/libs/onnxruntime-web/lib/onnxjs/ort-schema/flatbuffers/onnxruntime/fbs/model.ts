// automatically generated by the FlatBuffers compiler, do not modify

/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */

import * as flatbuffers from 'flatbuffers';

import { Graph } from './graph.js';
import { OperatorSetId } from './operator-set-id.js';
import { StringStringEntry } from './string-string-entry.js';

export class Model {
  bb: flatbuffers.ByteBuffer | null = null;
  bb_pos = 0;
  __init(i: number, bb: flatbuffers.ByteBuffer): Model {
    this.bb_pos = i;
    this.bb = bb;
    return this;
  }

  static getRootAsModel(bb: flatbuffers.ByteBuffer, obj?: Model): Model {
    return (obj || new Model()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  static getSizePrefixedRootAsModel(bb: flatbuffers.ByteBuffer, obj?: Model): Model {
    bb.setPosition(bb.position() + flatbuffers.SIZE_PREFIX_LENGTH);
    return (obj || new Model()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  irVersion(): bigint {
    const offset = this.bb!.__offset(this.bb_pos, 4);
    return offset ? this.bb!.readInt64(this.bb_pos + offset) : BigInt('0');
  }

  opsetImport(index: number, obj?: OperatorSetId): OperatorSetId | null {
    const offset = this.bb!.__offset(this.bb_pos, 6);
    return offset
      ? (obj || new OperatorSetId()).__init(
          this.bb!.__indirect(this.bb!.__vector(this.bb_pos + offset) + index * 4),
          this.bb!,
        )
      : null;
  }

  opsetImportLength(): number {
    const offset = this.bb!.__offset(this.bb_pos, 6);
    return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
  }

  producerName(): string | null;
  producerName(optionalEncoding: flatbuffers.Encoding): string | Uint8Array | null;
  producerName(optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 8);
    return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
  }

  producerVersion(): string | null;
  producerVersion(optionalEncoding: flatbuffers.Encoding): string | Uint8Array | null;
  producerVersion(optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 10);
    return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
  }

  domain(): string | null;
  domain(optionalEncoding: flatbuffers.Encoding): string | Uint8Array | null;
  domain(optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 12);
    return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
  }

  modelVersion(): bigint {
    const offset = this.bb!.__offset(this.bb_pos, 14);
    return offset ? this.bb!.readInt64(this.bb_pos + offset) : BigInt('0');
  }

  docString(): string | null;
  docString(optionalEncoding: flatbuffers.Encoding): string | Uint8Array | null;
  docString(optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 16);
    return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
  }

  graph(obj?: Graph): Graph | null {
    const offset = this.bb!.__offset(this.bb_pos, 18);
    return offset ? (obj || new Graph()).__init(this.bb!.__indirect(this.bb_pos + offset), this.bb!) : null;
  }

  graphDocString(): string | null;
  graphDocString(optionalEncoding: flatbuffers.Encoding): string | Uint8Array | null;
  graphDocString(optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 20);
    return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
  }

  metadataProps(index: number, obj?: StringStringEntry): StringStringEntry | null {
    const offset = this.bb!.__offset(this.bb_pos, 22);
    return offset
      ? (obj || new StringStringEntry()).__init(
          this.bb!.__indirect(this.bb!.__vector(this.bb_pos + offset) + index * 4),
          this.bb!,
        )
      : null;
  }

  metadataPropsLength(): number {
    const offset = this.bb!.__offset(this.bb_pos, 22);
    return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
  }

  static startModel(builder: flatbuffers.Builder) {
    builder.startObject(10);
  }

  static addIrVersion(builder: flatbuffers.Builder, irVersion: bigint) {
    builder.addFieldInt64(0, irVersion, BigInt('0'));
  }

  static addOpsetImport(builder: flatbuffers.Builder, opsetImportOffset: flatbuffers.Offset) {
    builder.addFieldOffset(1, opsetImportOffset, 0);
  }

  static createOpsetImportVector(builder: flatbuffers.Builder, data: flatbuffers.Offset[]): flatbuffers.Offset {
    builder.startVector(4, data.length, 4);
    for (let i = data.length - 1; i >= 0; i--) {
      builder.addOffset(data[i]!);
    }
    return builder.endVector();
  }

  static startOpsetImportVector(builder: flatbuffers.Builder, numElems: number) {
    builder.startVector(4, numElems, 4);
  }

  static addProducerName(builder: flatbuffers.Builder, producerNameOffset: flatbuffers.Offset) {
    builder.addFieldOffset(2, producerNameOffset, 0);
  }

  static addProducerVersion(builder: flatbuffers.Builder, producerVersionOffset: flatbuffers.Offset) {
    builder.addFieldOffset(3, producerVersionOffset, 0);
  }

  static addDomain(builder: flatbuffers.Builder, domainOffset: flatbuffers.Offset) {
    builder.addFieldOffset(4, domainOffset, 0);
  }

  static addModelVersion(builder: flatbuffers.Builder, modelVersion: bigint) {
    builder.addFieldInt64(5, modelVersion, BigInt('0'));
  }

  static addDocString(builder: flatbuffers.Builder, docStringOffset: flatbuffers.Offset) {
    builder.addFieldOffset(6, docStringOffset, 0);
  }

  static addGraph(builder: flatbuffers.Builder, graphOffset: flatbuffers.Offset) {
    builder.addFieldOffset(7, graphOffset, 0);
  }

  static addGraphDocString(builder: flatbuffers.Builder, graphDocStringOffset: flatbuffers.Offset) {
    builder.addFieldOffset(8, graphDocStringOffset, 0);
  }

  static addMetadataProps(builder: flatbuffers.Builder, metadataPropsOffset: flatbuffers.Offset) {
    builder.addFieldOffset(9, metadataPropsOffset, 0);
  }

  static createMetadataPropsVector(builder: flatbuffers.Builder, data: flatbuffers.Offset[]): flatbuffers.Offset {
    builder.startVector(4, data.length, 4);
    for (let i = data.length - 1; i >= 0; i--) {
      builder.addOffset(data[i]!);
    }
    return builder.endVector();
  }

  static startMetadataPropsVector(builder: flatbuffers.Builder, numElems: number) {
    builder.startVector(4, numElems, 4);
  }

  static endModel(builder: flatbuffers.Builder): flatbuffers.Offset {
    const offset = builder.endObject();
    return offset;
  }
}
