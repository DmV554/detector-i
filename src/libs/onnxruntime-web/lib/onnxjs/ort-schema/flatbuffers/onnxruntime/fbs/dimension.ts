// automatically generated by the FlatBuffers compiler, do not modify

/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */

import * as flatbuffers from 'flatbuffers';

import { DimensionValue } from './dimension-value.js';

export class Dimension {
  bb: flatbuffers.ByteBuffer | null = null;
  bb_pos = 0;
  __init(i: number, bb: flatbuffers.ByteBuffer): Dimension {
    this.bb_pos = i;
    this.bb = bb;
    return this;
  }

  static getRootAsDimension(bb: flatbuffers.ByteBuffer, obj?: Dimension): Dimension {
    return (obj || new Dimension()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  static getSizePrefixedRootAsDimension(bb: flatbuffers.ByteBuffer, obj?: Dimension): Dimension {
    bb.setPosition(bb.position() + flatbuffers.SIZE_PREFIX_LENGTH);
    return (obj || new Dimension()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  value(obj?: DimensionValue): DimensionValue | null {
    const offset = this.bb!.__offset(this.bb_pos, 4);
    return offset ? (obj || new DimensionValue()).__init(this.bb!.__indirect(this.bb_pos + offset), this.bb!) : null;
  }

  denotation(): string | null;
  denotation(optionalEncoding: flatbuffers.Encoding): string | Uint8Array | null;
  denotation(optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 6);
    return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
  }

  static startDimension(builder: flatbuffers.Builder) {
    builder.startObject(2);
  }

  static addValue(builder: flatbuffers.Builder, valueOffset: flatbuffers.Offset) {
    builder.addFieldOffset(0, valueOffset, 0);
  }

  static addDenotation(builder: flatbuffers.Builder, denotationOffset: flatbuffers.Offset) {
    builder.addFieldOffset(1, denotationOffset, 0);
  }

  static endDimension(builder: flatbuffers.Builder): flatbuffers.Offset {
    const offset = builder.endObject();
    return offset;
  }

  static createDimension(
    builder: flatbuffers.Builder,
    valueOffset: flatbuffers.Offset,
    denotationOffset: flatbuffers.Offset,
  ): flatbuffers.Offset {
    Dimension.startDimension(builder);
    Dimension.addValue(builder, valueOffset);
    Dimension.addDenotation(builder, denotationOffset);
    return Dimension.endDimension(builder);
  }
}
