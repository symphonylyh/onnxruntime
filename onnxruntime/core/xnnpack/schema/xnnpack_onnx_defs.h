// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <onnx/common/common.h>
#include <onnx/common/status.h>
#include <onnx/defs/schema.h>

#include "xnnpack_onnx_schema.h"
#ifndef ONNX_RETURN_IF_ERROR
#define ONNX_RETURN_IF_ERROR(expr) \
  do {                             \
    auto _status = (expr);         \
    if ((!_status.IsOK())) {       \
      return _status;              \
    }                              \
  } while (0)
#endif

namespace onnxruntime {
constexpr const char* kXNNPackDomain = "com.microsoft.xnnpack";
namespace xnnpack {
using OnnxStatus = ::ONNX_NAMESPACE::Common::Status;
// When this function returns OK, *output_size should equals to input_size whenever stride=1.
OnnxStatus ComputeOutputSizeSame(ptrdiff_t input_size, uint32_t stride, ptrdiff_t* output_size);
OnnxStatus ComputeOutputSizeValid(ptrdiff_t input_size, uint32_t stride, ptrdiff_t filter_size, uint32_t dilation_rate,
                                  ptrdiff_t* output_size);
OnnxStatus XnnPackConvShapeInferImpl(const ::ONNX_NAMESPACE::TensorShapeProto& input_shape,
                                     const ::ONNX_NAMESPACE::TensorShapeProto& weight_shape, uint32_t input_padding_top,
                                     uint32_t input_padding_right, uint32_t input_padding_bottom,
                                     uint32_t input_padding_left, uint32_t subsampling_height,
                                     uint32_t subsampling_width, uint32_t dilation_h, uint32_t dilation_w,
                                     int padding_mode, ::ONNX_NAMESPACE::TensorShapeProto* final_output_shape);
OnnxStatus XnnPackDepthwiseConvolution2dShapeInferImpl(const ::ONNX_NAMESPACE::TensorShapeProto& input_shape,
                                                       const ::ONNX_NAMESPACE::TensorShapeProto& weight_shape,
                                                       uint32_t input_padding_top, uint32_t input_padding_right,
                                                       uint32_t input_padding_bottom, uint32_t input_padding_left,
                                                       uint32_t subsampling_height, uint32_t subsampling_width,
                                                       uint32_t dilation_h, uint32_t dilation_w, int padding_mode,
                                                       ::ONNX_NAMESPACE::TensorShapeProto* final_output_shape);

inline void SetDimValue(::ONNX_NAMESPACE::TensorShapeProto_Dimension* left,
                        const ::ONNX_NAMESPACE::TensorShapeProto_Dimension& right) {
  *left = right;
}

inline void SetDimValue(::ONNX_NAMESPACE::TensorShapeProto_Dimension* left, int64_t right) {
  left->set_dim_value(right);
}

inline void SetDimValue(int64_t& left, int64_t right) { left = right; }

// padding_mode: 0, valid. 1, same
// T1: int64_t or ::ONNX_NAMESPACE::TensorShapeProto_Dimension
// T2: std::vector<int64> or std::array<int64,4> or std::array<::ONNX_NAMESPACE::TensorShapeProto_Dimension*, 4>
template <typename T1, typename T2>
OnnxStatus ConvShapeInference(const T1& batch_shape, ptrdiff_t in_height, ptrdiff_t in_width, ptrdiff_t in_channels,
                              const T1& out_channels, ptrdiff_t filter_height, ptrdiff_t filter_width,
                              ptrdiff_t in_channels1, uint32_t strides_h, uint32_t strides_w, uint32_t dilation_h,
                              uint32_t dilation_w, int padding_mode, T2& output) {
  if (in_channels != in_channels1) {
    return OnnxStatus(::ONNX_NAMESPACE::Common::StatusCategory::NONE, ::ONNX_NAMESPACE::Common::StatusCode::FAIL);
  }

  SetDimValue(output[0], batch_shape);
  ptrdiff_t output1, output2;
  if (padding_mode == 1) {
    ONNX_RETURN_IF_ERROR(ComputeOutputSizeSame(in_height, strides_h, &output1));
    ONNX_RETURN_IF_ERROR(ComputeOutputSizeSame(in_width, strides_w, &output2));
  } else if (padding_mode == 0) {
    ONNX_RETURN_IF_ERROR(ComputeOutputSizeValid(in_height, strides_h, filter_height, dilation_h, &output1));
    ONNX_RETURN_IF_ERROR(ComputeOutputSizeValid(in_width, strides_w, filter_width, dilation_w, &output2));
  } else {
    return OnnxStatus(::ONNX_NAMESPACE::Common::StatusCategory::NONE, ::ONNX_NAMESPACE::Common::StatusCode::FAIL,
                      "Invalid padding mode.");
  }

  SetDimValue(output[3], out_channels);
  SetDimValue(output[1], output1);
  SetDimValue(output[2], output2);
  return ::ONNX_NAMESPACE::Common::Status::OK();
}
}  // namespace xnnpack
}  // namespace onnxruntime

#define ONNX_XNNPACK_OPERATOR_SET_SCHEMA(name, ver, impl) \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, XnnPack, ::onnxruntime::kXNNPackDomain, ver, true, impl)
