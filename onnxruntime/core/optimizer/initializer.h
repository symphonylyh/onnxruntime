// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <functional>
#include <vector>
#include <cmath>

#include "core/common/common.h"
#include "core/common/path.h"
#include "core/framework/allocator.h"
#include "core/optimizer/graph_transformer.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"

namespace onnxruntime {

class Initializer final {
 public:
  // Construct an initializer with the provided name and data type, with all values initialized to 0
  Initializer(ONNX_NAMESPACE::TensorProto_DataType data_type,
              std::string_view name,
              gsl::span<const int64_t> dims);

  Initializer(const ONNX_NAMESPACE::TensorProto& tensor_proto,
              const Path& model_path);

  ~Initializer() = default;

  void ToProto(ONNX_NAMESPACE::TensorProto& tensor_proto) const {
    tensor_proto = utils::TensorToTensorProto(data_, name_);
  }

  ONNX_NAMESPACE::TensorProto ToFP16(const std::string& name) const;

  ONNX_NAMESPACE::TensorProto ToBFloat16(const std::string& name) const;

  int data_type() const {
    return data_.GetElementType();
  }

  std::string_view name() const {
    return name_;
  }

  template <typename T>
  T* data() {
    return data_.MutableData<T>();
  }

  template <typename T>
  const T* data() const {
    return data_.Data<T>();
  }

  gsl::span<const int64_t> dims() const {
    return data_.Shape().GetDims();
  }

  int64_t size() const { return data_.Shape().Size(); }

  Initializer& add(float value);

  Initializer& add(const Initializer& other);

  Initializer& sub(const Initializer& other);
  
  Initializer& mul(const Initializer& other);

  Initializer& div(const Initializer& other);

  Initializer& sqrt();

  void scale_by_axis(const Initializer& other, int axis);

 private:
//#if !defined(ORT_MINIMAL_BUILD)
//  static Status ReadExternalRawData(
//      const ONNX_NAMESPACE::TensorProto& tensor_proto, const Path& model_path, std::vector<char>& raw_data);
//#endif

  std::string name_;
  Tensor data_;
};

}  // namespace onnxruntime
