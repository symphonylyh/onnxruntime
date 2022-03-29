// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#include "core/optimizer/initializer.h"

#include "gsl/gsl"

#include "core/common/path.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/platform/env.h"

#include <functional>

namespace onnxruntime {

Initializer::Initializer(ONNX_NAMESPACE::TensorProto_DataType data_type,
                         std::string_view name,
                         gsl::span<const int64_t> dims) : name_(name) {
  Tensor w(DataTypeImpl::TensorTypeFromONNXEnum(data_type), dims, std::make_shared<CPUAllocator>());
  if (!w.IsDataTypeString()) {
    memset(w.MutableDataRaw(), 0, w.SizeInBytes());
  }
  data_ = std::move(w);
}

Initializer::Initializer(const ONNX_NAMESPACE::TensorProto& tensor_proto, const Path& model_path) {
  auto proto_data_type = tensor_proto.data_type();
  if (utils::HasName(tensor_proto)) {
    name_ = tensor_proto.name();
  }

  auto proto_dims = utils::GetTensorShapeFromTensorProto(tensor_proto);
  TensorShape proto_shape(proto_dims);

  // This must be pre-allocated
  Tensor w(DataTypeImpl::TensorTypeFromONNXEnum(proto_data_type), proto_shape, std::make_shared<CPUAllocator>());
  ORT_THROW_IF_ERROR(utils::TensorProtoToTensor(Env::Default(), model_path.ToPathString().c_str(), tensor_proto, w));
  data_ = std::move(w);
}

namespace {
template <typename T>
struct ToFp16;

template <>
struct ToFp16<MLFloat16> {
  uint16_t operator()(const MLFloat16& fl) const {
    return fl.val;
  }
};

template <>
struct ToFp16<BFloat16> {
  uint16_t operator()(const BFloat16& fl) const {
    return fl.val;
  }
};

template <>
struct ToFp16<float> {
  uint16_t operator()(float f) const {
    return math::floatToHalf(f);
  }
};

template <>
struct ToFp16<double> {
  uint16_t operator()(double d) const {
    return math::floatToHalf(static_cast<float>(d));
  }
};

template <typename T>
struct TensorToProtoFP16 {
  void operator()(const Tensor& data, ONNX_NAMESPACE::TensorProto& proto) const {
    ToFp16<T> to_fp16;
    auto span = data.DataAsSpan<T>();
    for (const auto& v : span) {
      proto.add_int32_data(to_fp16(v));
    }
  }
};

template <typename T>
struct ToBFloat16;

template <>
struct ToBFloat16<BFloat16> {
  uint16_t operator()(const BFloat16& bf) const {
    return bf.val;
  }
};

template <>
struct ToBFloat16<float> {
  uint16_t operator()(float f) const {
    return BFloat16(f).val;
  }
};

template <>
struct ToBFloat16<double> {
  uint16_t operator()(double d) const {
    return BFloat16(static_cast<float>(d)).val;
  }
};

template <typename T>
struct TensorToProtoBFloat16 {
  void operator()(const Tensor& data, ONNX_NAMESPACE::TensorProto& proto) const {
    ToBFloat16<T> to_bfloat16;
    auto span = data.DataAsSpan<T>();
    for (const auto& v : span) {
      proto.add_int32_data(to_bfloat16(v));
    }
  }
};

void SetNameDims(const std::string& name,
                 gsl::span<const int64_t> dims,
                 ONNX_NAMESPACE::TensorProto_DataType dt,
                 ONNX_NAMESPACE::TensorProto& tensor_proto) {
  tensor_proto.set_name(name);
  tensor_proto.set_data_type(dt);

  for (auto d : dims) {
    tensor_proto.add_dims(d);
  }
}

}  // namespace

ONNX_NAMESPACE::TensorProto Initializer::ToFP16(const std::string& name) const {
  ONNX_NAMESPACE::TensorProto tensor_proto;
  SetNameDims(name, data_.Shape().GetDims(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, tensor_proto);
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double> t_disp(data_.GetElementType());
  t_disp.Invoke<TensorToProtoFP16>(data_, tensor_proto);
  return tensor_proto;
}

ONNX_NAMESPACE::TensorProto Initializer::ToBFloat16(const std::string& name) const {
  ONNX_NAMESPACE::TensorProto tensor_proto;
  SetNameDims(name, data_.Shape().GetDims(), ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, tensor_proto);
  utils::MLTypeCallDispatcher<BFloat16, float, double> t_disp(data_.GetElementType());
  t_disp.Invoke<TensorToProtoBFloat16>(data_, tensor_proto);
  return tensor_proto;
}

namespace {

// Covers standard types
template <typename T>
struct ToNumeric {
  using type = T;
  constexpr const T& operator()(const T& v) const {
    return v;
  }
};

template <>
struct ToNumeric<MLFloat16> {
  using type = float;
  float operator()(const MLFloat16& v) const {
    return v.ToFloat();
  }
};

template <>
struct ToNumeric<BFloat16> {
  using type = float;
  float operator()(const BFloat16& v) const {
    return v.ToFloat();
  }
};

template <typename T, typename Op>
struct OpElementWise {
  void Invoke(Tensor& lhs, const Tensor& rhs) const {
    Op op;
    ToNumeric<T> to_numeric;
    auto dst_span = lhs.MutableDataAsSpan<T>();
    auto src_span = rhs.DataAsSpan<T>();
    ORT_ENFORCE(dst_span.size() <= src_span.size(), "RHS span must be at least as large as LHS span");
    for (size_t i = 0, limit = dst_span.size(); i < limit; ++i) {
      dst_span[i] = T(op(to_numeric(dst_span[i]), to_numeric(src_span[i])));
    }
  }
};

template <typename T>
struct ScalarAdd {
  void operator()(Tensor& tensor, float v) const {
    ToNumeric<T> to_numeric;
    auto span = tensor.MutableDataAsSpan<T>();
    for (auto& dst : span) {
      dst = T(to_numeric(dst) + v);
    }
  }
};

template <typename T>
struct Sqrt {
  void operator()(Tensor& tensor) const {
    ToNumeric<T> to_numeric;
    auto span = tensor.MutableDataAsSpan<T>();
    for (auto& dst : span) {
      auto v = to_numeric(dst);
      dst = T(v * v);
    }
  }
};

template <typename T>
struct ElementWiseAdd : OpElementWise<T, std::plus<typename ToNumeric<T>::type>> {
  void operator()(Tensor& lhs, const Tensor& rhs) const {
    Invoke(lhs, rhs);
  }
};

template <typename T>
struct ElementWiseSub : OpElementWise<T, std::minus<typename ToNumeric<T>::type>> {
  void operator()(Tensor& lhs, const Tensor& rhs) const {
    Invoke(lhs, rhs);
  }
};

template <typename T>
struct ElementWiseMul : OpElementWise<T, std::multiplies<typename ToNumeric<T>::type>> {
  void operator()(Tensor& lhs, const Tensor& rhs) const {
    Invoke(lhs, rhs);
  }
};

template <typename T>
struct ElementWiseDiv : OpElementWise<T, std::divides<typename ToNumeric<T>::type>> {
  void operator()(Tensor& lhs, const Tensor& rhs) const {
    Invoke(lhs, rhs);
  }
};
}  // namespace

Initializer&
Initializer::add(float value) {
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double> t_disp(data_.GetElementType());
  t_disp.Invoke<ScalarAdd>(data_, value);
  return *this;
}

Initializer& Initializer::add(const Initializer& other) {
  ORT_ENFORCE(data_type() == other.data_type(), "Expecting the same data type");
  ORT_ENFORCE(size() == other.size(), "Expecting the same size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_.GetElementType());
  t_disp.Invoke<ElementWiseAdd>(data_, other.data_);
  return *this;
}

Initializer& Initializer::sub(const Initializer& other) {
  ORT_ENFORCE(data_type() == other.data_type(), "Expecting the same data type");
  ORT_ENFORCE(size() == other.size(), "Expecting the same size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_.GetElementType());
  t_disp.Invoke<ElementWiseSub>(data_, other.data_);
  return *this;
}

Initializer& Initializer::mul(const Initializer& other) {
  ORT_ENFORCE(data_type() == other.data_type(), "Expecting the same data type");
  ORT_ENFORCE(size() == other.size(), "Expecting the same size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_.GetElementType());
  t_disp.Invoke<ElementWiseMul>(data_, other.data_);
  return *this;
}

Initializer& Initializer::div(const Initializer& other) {
  ORT_ENFORCE(data_type() == other.data_type(), "Expecting the same data type");
  ORT_ENFORCE(size() == other.size(), "Expecting the same size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_.GetElementType());
  t_disp.Invoke<ElementWiseDiv>(data_, other.data_);
  return *this;
}

Initializer& Initializer::sqrt() {
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double> t_disp(data_.GetElementType());
  t_disp.Invoke<Sqrt>(data_);
  return *this;
}

namespace {
template<typename T>
struct ScaleByAxis {
  void operator()(Tensor& data, const Tensor& scaler, const int64_t size_from_axis, const int64_t num_blocks) const {
    const auto scaler_size = scaler.Shape().Size();
    ToNumeric<T> to_numeric;
    T* dst = data.MutableData<T>();
    const T* src = scaler.Data<T>();
    for (size_t i = 0; i < static_cast<size_t>(num_blocks); i++) {
      size_t scaler_index = scaler_size == 1 ? 0 : i;
      for (size_t j = 0, limit = static_cast<size_t>(size_from_axis); j < limit; ++j) {
        auto k = i * size_from_axis + j;
        dst[k] = T(to_numeric(dst[k]) * to_numeric(src[scaler_index]));
      }
    }
  }
};

}

void Initializer::scale_by_axis(const Initializer& other, int axis) {
  const int64_t size_from_axis = data_.Shape().SizeFromDimension(gsl::narrow_cast<size_t>(axis));
  const int64_t num_blocks = size() / size_from_axis;
  ORT_ENFORCE(other.size() == 1 || other.size() == num_blocks, "Invalid other(scalers) size");
  utils::MLTypeCallDispatcher<MLFloat16, BFloat16, float, double, int32_t, int64_t> t_disp(data_.GetElementType());
  t_disp.Invoke<ScaleByAxis>(data_, other.data_, size_from_axis, num_blocks);
}

//Status Initializer::ReadExternalRawData(
//    const ONNX_NAMESPACE::TensorProto& tensor_proto, const Path& model_path, std::vector<char>& raw_data) {
//  ORT_RETURN_IF_NOT(
//      tensor_proto.data_type() != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED &&
//          tensor_proto.data_type() != ONNX_NAMESPACE::TensorProto_DataType_STRING,
//      "External data type must not be UNDEFINED or STRING.");
//
//  ORT_RETURN_IF(
//      model_path.IsEmpty(),
//      "model_path must not be empty. Ensure that a path is provided when the model is created or loaded.");
//
//  std::unique_ptr<ExternalDataInfo> external_data{};
//  ORT_RETURN_IF_ERROR(ExternalDataInfo::Create(tensor_proto.external_data(), external_data));
//
//  size_t actual_tensor_data_length;
//  ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProto<0>(
//      tensor_proto, &actual_tensor_data_length));
//  const size_t external_data_length = external_data->GetLength();
//
//  ORT_RETURN_IF_NOT(
//      external_data_length == 0 ||
//          external_data_length == actual_tensor_data_length,
//      "TensorProto external data size mismatch. ",
//      "Computed size: ", actual_tensor_data_length,
//      ", external_data.length: ", external_data_length);
//
//  Path external_data_relative_path{};
//  ORT_RETURN_IF_ERROR(Path::Parse(
//      external_data->GetRelPath(), external_data_relative_path));
//
//  std::vector<char> buffer(actual_tensor_data_length);
//
//  ORT_RETURN_IF_ERROR(Env::Default().ReadFileIntoBuffer(
//      (model_path.ParentPath() / external_data_relative_path).ToPathString().c_str(),
//      external_data->GetOffset(),
//      actual_tensor_data_length,
//      gsl::make_span(buffer)));
//
//  raw_data = std::move(buffer);
//
//  return Status::OK();
//}
}  // namespace onnxruntime

#endif  // !(ORT_MINIMAL_BUILD)
