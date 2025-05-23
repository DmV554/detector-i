'use strict';
// automatically generated by the FlatBuffers compiler, do not modify
Object.defineProperty(exports, '__esModule', { value: true });
exports.ValueInfo =
  exports.TypeInfoValue =
  exports.TypeInfo =
  exports.TensorTypeAndShape =
  exports.TensorDataType =
  exports.Tensor =
  exports.StringStringEntry =
  exports.SparseTensor =
  exports.Shape =
  exports.SequenceType =
  exports.RuntimeOptimizations =
  exports.RuntimeOptimizationRecordContainerEntry =
  exports.RuntimeOptimizationRecord =
  exports.OperatorSetId =
  exports.OpIdKernelTypeStrArgsEntry =
  exports.NodesToOptimizeIndices =
  exports.NodeType =
  exports.NodeEdge =
  exports.Node =
  exports.Model =
  exports.MapType =
  exports.KernelTypeStrResolver =
  exports.KernelTypeStrArgsEntry =
  exports.InferenceSession =
  exports.Graph =
  exports.EdgeEnd =
  exports.DimensionValueType =
  exports.DimensionValue =
  exports.Dimension =
  exports.DeprecatedSubGraphSessionState =
  exports.DeprecatedSessionState =
  exports.DeprecatedNodeIndexAndKernelDefHash =
  exports.DeprecatedKernelCreateInfos =
  exports.AttributeType =
  exports.Attribute =
  exports.ArgTypeAndIndex =
  exports.ArgType =
    void 0;
/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-explicit-any, @typescript-eslint/no-non-null-assertion */
var arg_type_js_1 = require('./fbs/arg-type.js');
Object.defineProperty(exports, 'ArgType', {
  enumerable: true,
  get: function () {
    return arg_type_js_1.ArgType;
  },
});
var arg_type_and_index_js_1 = require('./fbs/arg-type-and-index.js');
Object.defineProperty(exports, 'ArgTypeAndIndex', {
  enumerable: true,
  get: function () {
    return arg_type_and_index_js_1.ArgTypeAndIndex;
  },
});
var attribute_js_1 = require('./fbs/attribute.js');
Object.defineProperty(exports, 'Attribute', {
  enumerable: true,
  get: function () {
    return attribute_js_1.Attribute;
  },
});
var attribute_type_js_1 = require('./fbs/attribute-type.js');
Object.defineProperty(exports, 'AttributeType', {
  enumerable: true,
  get: function () {
    return attribute_type_js_1.AttributeType;
  },
});
var deprecated_kernel_create_infos_js_1 = require('./fbs/deprecated-kernel-create-infos.js');
Object.defineProperty(exports, 'DeprecatedKernelCreateInfos', {
  enumerable: true,
  get: function () {
    return deprecated_kernel_create_infos_js_1.DeprecatedKernelCreateInfos;
  },
});
var deprecated_node_index_and_kernel_def_hash_js_1 = require('./fbs/deprecated-node-index-and-kernel-def-hash.js');
Object.defineProperty(exports, 'DeprecatedNodeIndexAndKernelDefHash', {
  enumerable: true,
  get: function () {
    return deprecated_node_index_and_kernel_def_hash_js_1.DeprecatedNodeIndexAndKernelDefHash;
  },
});
var deprecated_session_state_js_1 = require('./fbs/deprecated-session-state.js');
Object.defineProperty(exports, 'DeprecatedSessionState', {
  enumerable: true,
  get: function () {
    return deprecated_session_state_js_1.DeprecatedSessionState;
  },
});
var deprecated_sub_graph_session_state_js_1 = require('./fbs/deprecated-sub-graph-session-state.js');
Object.defineProperty(exports, 'DeprecatedSubGraphSessionState', {
  enumerable: true,
  get: function () {
    return deprecated_sub_graph_session_state_js_1.DeprecatedSubGraphSessionState;
  },
});
var dimension_js_1 = require('./fbs/dimension.js');
Object.defineProperty(exports, 'Dimension', {
  enumerable: true,
  get: function () {
    return dimension_js_1.Dimension;
  },
});
var dimension_value_js_1 = require('./fbs/dimension-value.js');
Object.defineProperty(exports, 'DimensionValue', {
  enumerable: true,
  get: function () {
    return dimension_value_js_1.DimensionValue;
  },
});
var dimension_value_type_js_1 = require('./fbs/dimension-value-type.js');
Object.defineProperty(exports, 'DimensionValueType', {
  enumerable: true,
  get: function () {
    return dimension_value_type_js_1.DimensionValueType;
  },
});
var edge_end_js_1 = require('./fbs/edge-end.js');
Object.defineProperty(exports, 'EdgeEnd', {
  enumerable: true,
  get: function () {
    return edge_end_js_1.EdgeEnd;
  },
});
var graph_js_1 = require('./fbs/graph.js');
Object.defineProperty(exports, 'Graph', {
  enumerable: true,
  get: function () {
    return graph_js_1.Graph;
  },
});
var inference_session_js_1 = require('./fbs/inference-session.js');
Object.defineProperty(exports, 'InferenceSession', {
  enumerable: true,
  get: function () {
    return inference_session_js_1.InferenceSession;
  },
});
var kernel_type_str_args_entry_js_1 = require('./fbs/kernel-type-str-args-entry.js');
Object.defineProperty(exports, 'KernelTypeStrArgsEntry', {
  enumerable: true,
  get: function () {
    return kernel_type_str_args_entry_js_1.KernelTypeStrArgsEntry;
  },
});
var kernel_type_str_resolver_js_1 = require('./fbs/kernel-type-str-resolver.js');
Object.defineProperty(exports, 'KernelTypeStrResolver', {
  enumerable: true,
  get: function () {
    return kernel_type_str_resolver_js_1.KernelTypeStrResolver;
  },
});
var map_type_js_1 = require('./fbs/map-type.js');
Object.defineProperty(exports, 'MapType', {
  enumerable: true,
  get: function () {
    return map_type_js_1.MapType;
  },
});
var model_js_1 = require('./fbs/model.js');
Object.defineProperty(exports, 'Model', {
  enumerable: true,
  get: function () {
    return model_js_1.Model;
  },
});
var node_js_1 = require('./fbs/node.js');
Object.defineProperty(exports, 'Node', {
  enumerable: true,
  get: function () {
    return node_js_1.Node;
  },
});
var node_edge_js_1 = require('./fbs/node-edge.js');
Object.defineProperty(exports, 'NodeEdge', {
  enumerable: true,
  get: function () {
    return node_edge_js_1.NodeEdge;
  },
});
var node_type_js_1 = require('./fbs/node-type.js');
Object.defineProperty(exports, 'NodeType', {
  enumerable: true,
  get: function () {
    return node_type_js_1.NodeType;
  },
});
var nodes_to_optimize_indices_js_1 = require('./fbs/nodes-to-optimize-indices.js');
Object.defineProperty(exports, 'NodesToOptimizeIndices', {
  enumerable: true,
  get: function () {
    return nodes_to_optimize_indices_js_1.NodesToOptimizeIndices;
  },
});
var op_id_kernel_type_str_args_entry_js_1 = require('./fbs/op-id-kernel-type-str-args-entry.js');
Object.defineProperty(exports, 'OpIdKernelTypeStrArgsEntry', {
  enumerable: true,
  get: function () {
    return op_id_kernel_type_str_args_entry_js_1.OpIdKernelTypeStrArgsEntry;
  },
});
var operator_set_id_js_1 = require('./fbs/operator-set-id.js');
Object.defineProperty(exports, 'OperatorSetId', {
  enumerable: true,
  get: function () {
    return operator_set_id_js_1.OperatorSetId;
  },
});
var runtime_optimization_record_js_1 = require('./fbs/runtime-optimization-record.js');
Object.defineProperty(exports, 'RuntimeOptimizationRecord', {
  enumerable: true,
  get: function () {
    return runtime_optimization_record_js_1.RuntimeOptimizationRecord;
  },
});
var runtime_optimization_record_container_entry_js_1 = require('./fbs/runtime-optimization-record-container-entry.js');
Object.defineProperty(exports, 'RuntimeOptimizationRecordContainerEntry', {
  enumerable: true,
  get: function () {
    return runtime_optimization_record_container_entry_js_1.RuntimeOptimizationRecordContainerEntry;
  },
});
var runtime_optimizations_js_1 = require('./fbs/runtime-optimizations.js');
Object.defineProperty(exports, 'RuntimeOptimizations', {
  enumerable: true,
  get: function () {
    return runtime_optimizations_js_1.RuntimeOptimizations;
  },
});
var sequence_type_js_1 = require('./fbs/sequence-type.js');
Object.defineProperty(exports, 'SequenceType', {
  enumerable: true,
  get: function () {
    return sequence_type_js_1.SequenceType;
  },
});
var shape_js_1 = require('./fbs/shape.js');
Object.defineProperty(exports, 'Shape', {
  enumerable: true,
  get: function () {
    return shape_js_1.Shape;
  },
});
var sparse_tensor_js_1 = require('./fbs/sparse-tensor.js');
Object.defineProperty(exports, 'SparseTensor', {
  enumerable: true,
  get: function () {
    return sparse_tensor_js_1.SparseTensor;
  },
});
var string_string_entry_js_1 = require('./fbs/string-string-entry.js');
Object.defineProperty(exports, 'StringStringEntry', {
  enumerable: true,
  get: function () {
    return string_string_entry_js_1.StringStringEntry;
  },
});
var tensor_js_1 = require('./fbs/tensor.js');
Object.defineProperty(exports, 'Tensor', {
  enumerable: true,
  get: function () {
    return tensor_js_1.Tensor;
  },
});
var tensor_data_type_js_1 = require('./fbs/tensor-data-type.js');
Object.defineProperty(exports, 'TensorDataType', {
  enumerable: true,
  get: function () {
    return tensor_data_type_js_1.TensorDataType;
  },
});
var tensor_type_and_shape_js_1 = require('./fbs/tensor-type-and-shape.js');
Object.defineProperty(exports, 'TensorTypeAndShape', {
  enumerable: true,
  get: function () {
    return tensor_type_and_shape_js_1.TensorTypeAndShape;
  },
});
var type_info_js_1 = require('./fbs/type-info.js');
Object.defineProperty(exports, 'TypeInfo', {
  enumerable: true,
  get: function () {
    return type_info_js_1.TypeInfo;
  },
});
var type_info_value_js_1 = require('./fbs/type-info-value.js');
Object.defineProperty(exports, 'TypeInfoValue', {
  enumerable: true,
  get: function () {
    return type_info_value_js_1.TypeInfoValue;
  },
});
var value_info_js_1 = require('./fbs/value-info.js');
Object.defineProperty(exports, 'ValueInfo', {
  enumerable: true,
  get: function () {
    return value_info_js_1.ValueInfo;
  },
});
//# sourceMappingURL=fbs.js.map
