// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/contrib_ops/nhwc_inference_context.h"

void onnxruntime::contrib::NhwcInferenceContext::TransposeInputShape() {
  const auto* nhwc_type = ctx_.getInputType(0);
  if (nhwc_type != nullptr && nhwc_type->tensor_type().has_shape()) {
    const auto& nhwc_shape = nhwc_type->tensor_type().shape();
    const int rank = nhwc_shape.dim_size();
    if (rank < 2) {
      fail_shape_inference("Input tensor must have at least 2 dimensions");
    }
    // Convert input shape from {N, H, W, C} to {N, C, H, w}.
    auto* nchw_shape = input_type_.mutable_tensor_type()->mutable_shape();
    *nchw_shape->add_dim() = nhwc_shape.dim(0);
    *nchw_shape->add_dim() = nhwc_shape.dim(rank - 1);
    for (int i = 1; i < rank - 1; i++) {
      *nchw_shape->add_dim() = nhwc_shape.dim(i);
    }
  }
}

void onnxruntime::contrib::NhwcInferenceContext::TransposeOutputShape() {
  if (output_type_.tensor_type().has_shape()) {
    const auto& nchw_shape = output_type_.tensor_type().shape();
    const int rank = nchw_shape.dim_size();
    if (rank < 2) {
      fail_shape_inference("Output tensor must have at least 2 dimensions");
    }
    // Convert output shape from {N, C, H, W} to {N, H, w, C}.
    auto* nhwc_shape = ctx_.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    *nhwc_shape->add_dim() = nchw_shape.dim(0);
    for (int i = 2; i < rank; i++) {
      *nhwc_shape->add_dim() = nchw_shape.dim(i);
    }
    *nhwc_shape->add_dim() = nchw_shape.dim(1);
  }
}
