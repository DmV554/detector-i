// automatically generated by the FlatBuffers compiler, do not modify

/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */

import * as flatbuffers from 'flatbuffers';

import { TensorDataType } from './tensor-data-type.js';
import { TypeInfo } from './type-info.js';

export class MapType {
  bb: flatbuffers.ByteBuffer | null = null;
  bb_pos = 0;
  __init(i: number, bb: flatbuffers.ByteBuffer): MapType {
    this.bb_pos = i;
    this.bb = bb;
    return this;
  }

  static getRootAsMapType(bb: flatbuffers.ByteBuffer, obj?: MapType): MapType {
    return (obj || new MapType()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  static getSizePrefixedRootAsMapType(bb: flatbuffers.ByteBuffer, obj?: MapType): MapType {
    bb.setPosition(bb.position() + flatbuffers.SIZE_PREFIX_LENGTH);
    return (obj || new MapType()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  keyType(): TensorDataType {
    const offset = this.bb!.__offset(this.bb_pos, 4);
    return offset ? this.bb!.readInt32(this.bb_pos + offset) : TensorDataType.UNDEFINED;
  }

  valueType(obj?: TypeInfo): TypeInfo | null {
    const offset = this.bb!.__offset(this.bb_pos, 6);
    return offset ? (obj || new TypeInfo()).__init(this.bb!.__indirect(this.bb_pos + offset), this.bb!) : null;
  }

  static startMapType(builder: flatbuffers.Builder) {
    builder.startObject(2);
  }

  static addKeyType(builder: flatbuffers.Builder, keyType: TensorDataType) {
    builder.addFieldInt32(0, keyType, TensorDataType.UNDEFINED);
  }

  static addValueType(builder: flatbuffers.Builder, valueTypeOffset: flatbuffers.Offset) {
    builder.addFieldOffset(1, valueTypeOffset, 0);
  }

  static endMapType(builder: flatbuffers.Builder): flatbuffers.Offset {
    const offset = builder.endObject();
    return offset;
  }
}
