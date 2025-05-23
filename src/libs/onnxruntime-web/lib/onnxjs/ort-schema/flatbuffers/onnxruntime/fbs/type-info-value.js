'use strict';
// automatically generated by the FlatBuffers compiler, do not modify
Object.defineProperty(exports, '__esModule', { value: true });
exports.unionListToTypeInfoValue = exports.unionToTypeInfoValue = exports.TypeInfoValue = void 0;
/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */
const map_type_js_1 = require('./map-type.js');
const sequence_type_js_1 = require('./sequence-type.js');
const tensor_type_and_shape_js_1 = require('./tensor-type-and-shape.js');
var TypeInfoValue;
(function (TypeInfoValue) {
  TypeInfoValue[(TypeInfoValue['NONE'] = 0)] = 'NONE';
  TypeInfoValue[(TypeInfoValue['tensor_type'] = 1)] = 'tensor_type';
  TypeInfoValue[(TypeInfoValue['sequence_type'] = 2)] = 'sequence_type';
  TypeInfoValue[(TypeInfoValue['map_type'] = 3)] = 'map_type';
})(TypeInfoValue || (exports.TypeInfoValue = TypeInfoValue = {}));
function unionToTypeInfoValue(type, accessor) {
  switch (TypeInfoValue[type]) {
    case 'NONE':
      return null;
    case 'tensor_type':
      return accessor(new tensor_type_and_shape_js_1.TensorTypeAndShape());
    case 'sequence_type':
      return accessor(new sequence_type_js_1.SequenceType());
    case 'map_type':
      return accessor(new map_type_js_1.MapType());
    default:
      return null;
  }
}
exports.unionToTypeInfoValue = unionToTypeInfoValue;
function unionListToTypeInfoValue(type, accessor, index) {
  switch (TypeInfoValue[type]) {
    case 'NONE':
      return null;
    case 'tensor_type':
      return accessor(index, new tensor_type_and_shape_js_1.TensorTypeAndShape());
    case 'sequence_type':
      return accessor(index, new sequence_type_js_1.SequenceType());
    case 'map_type':
      return accessor(index, new map_type_js_1.MapType());
    default:
      return null;
  }
}
exports.unionListToTypeInfoValue = unionListToTypeInfoValue;
//# sourceMappingURL=type-info-value.js.map
