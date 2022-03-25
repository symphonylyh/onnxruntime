// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <onnx/defs/shape_inference.h>
#include <onnx/defs/schema.h>
#include <core/graph/constants.h>

namespace onnxruntime {
namespace contrib {
class NhwcInferenceContext : public ::ONNX_NAMESPACE::InferenceContext {
 public:
  NhwcInferenceContext(::ONNX_NAMESPACE::InferenceContext& ctx) : ctx_(ctx) {
  }

  void TransposeInputShape();
  void TransposeOutputShape();

  bool propagateElemTypeToInput(const ::ONNX_NAMESPACE::TypeProto* src) {
    return propagateElemType(src, &input_type_);
  }
  bool propagateElemTypeToOutput(::ONNX_NAMESPACE::TypeProto* dest) {
    return propagateElemType(&output_type_, dest);
  }

 protected:
  const ::ONNX_NAMESPACE::AttributeProto* getAttribute(const std::string& name) const override {
    return ctx_.getAttribute(name);
  }

  size_t getNumInputs() const noexcept override {
    return ctx_.getNumInputs();
  }

  const ::ONNX_NAMESPACE::TypeProto* getInputType(size_t index) const override {
    return (index == 0) ? &input_type_ : ctx_.getInputType(index);
  }

  const ::ONNX_NAMESPACE::TensorProto* getInputData(size_t) const override {
    return nullptr;
  }

  size_t getNumOutputs() const noexcept override {
    return ctx_.getNumOutputs();
  }

  ::ONNX_NAMESPACE::TypeProto* getOutputType(size_t index) override {
    return (index == 0) ? &output_type_ : ctx_.getOutputType(index);
  }

  ::ONNX_NAMESPACE::GraphInferencer* getGraphAttributeInferencer(const std::string&) override {
    return nullptr;
  }

  const ::ONNX_NAMESPACE::SparseTensorProto* getInputSparseData(size_t) const override {
    return nullptr;
  }

  const ::ONNX_NAMESPACE::TensorShapeProto* getSymbolicInput(size_t) const override {
    return nullptr;
  }

  static bool propagateElemType(const ::ONNX_NAMESPACE::TypeProto* input_type, ::ONNX_NAMESPACE::TypeProto* output_type) {
    const auto input_value_case = input_type->value_case();
    if (input_value_case != ::ONNX_NAMESPACE::TypeProto::kTensorType && input_value_case != ::ONNX_NAMESPACE::TypeProto::kSparseTensorType) {
      return false;
    }

    const auto input_elem_type = getTensorElementType(*input_type);
    if (input_elem_type == ::ONNX_NAMESPACE::TensorProto::UNDEFINED) {
      return false;
    }
    const auto output_value_case = output_type->value_case();
    if (output_value_case == ::ONNX_NAMESPACE::TypeProto::kTensorType || output_value_case == ::ONNX_NAMESPACE::TypeProto::kSparseTensorType) {
      setTensorElementType(input_elem_type, output_value_case, *output_type);
    } else if (output_value_case == ::ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET) {
      // Assume output will have the same type
      setTensorElementType(input_elem_type, input_value_case, *output_type);
    } else {
      return false;
    }
    return true;
  }

 private:
  ::ONNX_NAMESPACE::InferenceContext& ctx_;
  ::ONNX_NAMESPACE::TypeProto input_type_;
  ::ONNX_NAMESPACE::TypeProto output_type_;
};

struct NHWCInference {
  ::ONNX_NAMESPACE::InferenceFunction func_;

  void operator()(::ONNX_NAMESPACE::InferenceContext& context) const {
    NhwcInferenceContext ctx(context);
    if (!ctx.propagateElemTypeToInput(context.getInputType(0))) {
      fail_type_inference("inputs are expected to have tensor type.");
    }
    ctx.TransposeInputShape();
    func_(ctx);
    ctx.TransposeOutputShape();
    if (!ctx.propagateElemTypeToOutput(context.getOutputType(0))) {
      fail_type_inference("inputs are expected to have tensor type.");
    }
  }
};

template <typename F>
void RegisterNHWCSchema(F&& f, const ::ONNX_NAMESPACE::OpSchema& schema) {
  f(std::move(::ONNX_NAMESPACE::OpSchema(schema).TypeAndShapeInferenceFunction(NHWCInference{schema.GetTypeAndShapeInferenceFunction()}).SetDomain(onnxruntime::kMSInternalNHWCDomain)));
}
}  // namespace contrib
}  // namespace onnxruntime