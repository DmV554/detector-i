// automatically generated by the FlatBuffers compiler, do not modify

/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */

import * as flatbuffers from 'flatbuffers';

import { KernelTypeStrArgsEntry } from './kernel-type-str-args-entry.js';

export class OpIdKernelTypeStrArgsEntry {
  bb: flatbuffers.ByteBuffer | null = null;
  bb_pos = 0;
  __init(i: number, bb: flatbuffers.ByteBuffer): OpIdKernelTypeStrArgsEntry {
    this.bb_pos = i;
    this.bb = bb;
    return this;
  }

  static getRootAsOpIdKernelTypeStrArgsEntry(
    bb: flatbuffers.ByteBuffer,
    obj?: OpIdKernelTypeStrArgsEntry,
  ): OpIdKernelTypeStrArgsEntry {
    return (obj || new OpIdKernelTypeStrArgsEntry()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  static getSizePrefixedRootAsOpIdKernelTypeStrArgsEntry(
    bb: flatbuffers.ByteBuffer,
    obj?: OpIdKernelTypeStrArgsEntry,
  ): OpIdKernelTypeStrArgsEntry {
    bb.setPosition(bb.position() + flatbuffers.SIZE_PREFIX_LENGTH);
    return (obj || new OpIdKernelTypeStrArgsEntry()).__init(bb.readInt32(bb.position()) + bb.position(), bb);
  }

  opId(): string | null;
  opId(optionalEncoding: flatbuffers.Encoding): string | Uint8Array | null;
  opId(optionalEncoding?: any): string | Uint8Array | null {
    const offset = this.bb!.__offset(this.bb_pos, 4);
    return offset ? this.bb!.__string(this.bb_pos + offset, optionalEncoding) : null;
  }

  kernelTypeStrArgs(index: number, obj?: KernelTypeStrArgsEntry): KernelTypeStrArgsEntry | null {
    const offset = this.bb!.__offset(this.bb_pos, 6);
    return offset
      ? (obj || new KernelTypeStrArgsEntry()).__init(
          this.bb!.__indirect(this.bb!.__vector(this.bb_pos + offset) + index * 4),
          this.bb!,
        )
      : null;
  }

  kernelTypeStrArgsLength(): number {
    const offset = this.bb!.__offset(this.bb_pos, 6);
    return offset ? this.bb!.__vector_len(this.bb_pos + offset) : 0;
  }

  static startOpIdKernelTypeStrArgsEntry(builder: flatbuffers.Builder) {
    builder.startObject(2);
  }

  static addOpId(builder: flatbuffers.Builder, opIdOffset: flatbuffers.Offset) {
    builder.addFieldOffset(0, opIdOffset, 0);
  }

  static addKernelTypeStrArgs(builder: flatbuffers.Builder, kernelTypeStrArgsOffset: flatbuffers.Offset) {
    builder.addFieldOffset(1, kernelTypeStrArgsOffset, 0);
  }

  static createKernelTypeStrArgsVector(builder: flatbuffers.Builder, data: flatbuffers.Offset[]): flatbuffers.Offset {
    builder.startVector(4, data.length, 4);
    for (let i = data.length - 1; i >= 0; i--) {
      builder.addOffset(data[i]!);
    }
    return builder.endVector();
  }

  static startKernelTypeStrArgsVector(builder: flatbuffers.Builder, numElems: number) {
    builder.startVector(4, numElems, 4);
  }

  static endOpIdKernelTypeStrArgsEntry(builder: flatbuffers.Builder): flatbuffers.Offset {
    const offset = builder.endObject();
    builder.requiredField(offset, 4); // op_id
    return offset;
  }

  static createOpIdKernelTypeStrArgsEntry(
    builder: flatbuffers.Builder,
    opIdOffset: flatbuffers.Offset,
    kernelTypeStrArgsOffset: flatbuffers.Offset,
  ): flatbuffers.Offset {
    OpIdKernelTypeStrArgsEntry.startOpIdKernelTypeStrArgsEntry(builder);
    OpIdKernelTypeStrArgsEntry.addOpId(builder, opIdOffset);
    OpIdKernelTypeStrArgsEntry.addKernelTypeStrArgs(builder, kernelTypeStrArgsOffset);
    return OpIdKernelTypeStrArgsEntry.endOpIdKernelTypeStrArgsEntry(builder);
  }
}
