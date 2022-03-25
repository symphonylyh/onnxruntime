// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)
#include "core/optimizer/initializer.h"

#include "gsl/gsl"

#include "core/common/path.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/session_options.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/platform/env.h"

namespace onnxruntime {

Initializer::Initializer(const ONNX_NAMESPACE::TensorProto& tensor_proto, const DataTransferManager& dt_manager,
                         const SessionOptions& sess_options, const Path& model_path) {
  data_type_ = tensor_proto.data_type();
  if (utils::HasName(tensor_proto)) {
    name_ = tensor_proto.name();
  }
  dims_.reserve(tensor_proto.dims_size());
  for (int i = 0; i < tensor_proto.dims_size(); i++) {
    dims_.push_back(tensor_proto.dims(i));
  }

  size_ = std::accumulate(dims_.begin(), dims_.end(), static_cast<int64_t>(1), std::multiplies<int64_t>{});
  // Check if the initializer is among shared initializers
  if (!name_.empty()) {
    auto hit = sess_options.initializers_to_share_map.find(name_);
    if (hit != sess_options.initializers_to_share_map.end()) {
      const OrtValue& ort_value = *hit->second;
      const auto& tensor = ort_value.Get<Tensor>();
      ORT_ENFORCE(data_type_ == tensor.GetElementType(), "Element types do not agree for this initializer: ", name_);
      const auto w_dims = tensor.Shape().GetDims();
      ORT_ENFORCE(w_dims == gsl::make_span(dims_), "Dims do not agree for this initializer: ", name_);

      const auto size_in_bytes = tensor.SizeInBytes();
      raw_data_.resize(size_in_bytes);
      Tensor dst_tensor(tensor.DataType(), tensor.Shape(), raw_data_.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtArenaAllocator));
      ORT_THROW_IF_ERROR(dt_manager.CopyTensor(tensor, dst_tensor));
      return;
    }
  }

  if (tensor_proto.data_location() != ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
    if (utils::HasRawData(tensor_proto)) {
      raw_data_.assign(tensor_proto.raw_data().begin(), tensor_proto.raw_data().end());
    } else {
      switch (data_type_) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16: {
          int64_t size = tensor_proto.int32_data_size();
          ORT_ENFORCE(size_ == size, "size is different");
          for (int i = 0; i < size_; i++) {
            float16_data_.push_back(static_cast<uint16_t>(tensor_proto.int32_data(i)));
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
          int64_t size = tensor_proto.float_data_size();
          ORT_ENFORCE(size_ == size, "size is different");
          for (int i = 0; i < size_; i++) {
            float_data_.push_back(tensor_proto.float_data(i));
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
          int64_t size = tensor_proto.double_data_size();
          ORT_ENFORCE(size_ == size, "size is different");
          for (int i = 0; i < size_; i++) {
            double_data_.push_back(tensor_proto.double_data(i));
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
          int64_t size = tensor_proto.int32_data_size();
          ORT_ENFORCE(size_ == size, "size is different");
          for (int i = 0; i < size_; i++) {
            int8_data_.push_back(static_cast<int8_t>(tensor_proto.int32_data(i)));
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
          int64_t size = tensor_proto.int32_data_size();
          ORT_ENFORCE(size_ == size, "size is different");
          for (int i = 0; i < size_; i++) {
            uint8_data_.push_back(static_cast<uint8_t>(tensor_proto.int32_data(i)));
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
          int64_t size = tensor_proto.int32_data_size();
          ORT_ENFORCE(size_ == size, "size is different");
          for (int i = 0; i < size_; i++) {
            int32_data_.push_back(tensor_proto.int32_data(i));
          }
          break;
        }
        case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
          int64_t size = tensor_proto.int64_data_size();
          ORT_ENFORCE(size_ == size, "size is different");
          for (int i = 0; i < size_; i++) {
            int64_data_.push_back(tensor_proto.int64_data(i));
          }
          break;
        }
        default:
          ORT_NOT_IMPLEMENTED(__FUNCTION__, "unsupported data type: ", data_type_);
          break;
      }
    }
  } else {  // tensor_proto.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL
#if !defined(ORT_MINIMAL_BUILD)
    const auto status = ReadExternalRawData(tensor_proto, model_path, raw_data_);
    ORT_ENFORCE(status.IsOK(), "ReadExternalRawData() failed: ", status.ErrorMessage());
#else
    ORT_UNUSED_PARAMETER(model_path);
    ORT_THROW("External data is not supported in an ORT formal model.");
#endif
  }
}

Status Initializer::ReadExternalRawData(
    const ONNX_NAMESPACE::TensorProto& tensor_proto, const Path& model_path, std::vector<char>& raw_data) {
  ORT_RETURN_IF_NOT(
      tensor_proto.data_type() != ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED &&
          tensor_proto.data_type() != ONNX_NAMESPACE::TensorProto_DataType_STRING,
      "External data type must not be UNDEFINED or STRING.");

  ORT_RETURN_IF(
      model_path.IsEmpty(),
      "model_path must not be empty. Ensure that a path is provided when the model is created or loaded.");

  std::unique_ptr<ExternalDataInfo> external_data{};
  ORT_RETURN_IF_ERROR(ExternalDataInfo::Create(tensor_proto.external_data(), external_data));

  size_t actual_tensor_data_length;
  ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProto<0>(
      tensor_proto, &actual_tensor_data_length));
  const size_t external_data_length = external_data->GetLength();

  ORT_RETURN_IF_NOT(
      external_data_length == 0 ||
          external_data_length == actual_tensor_data_length,
      "TensorProto external data size mismatch. ",
      "Computed size: ", actual_tensor_data_length,
      ", external_data.length: ", external_data_length);

  Path external_data_relative_path{};
  ORT_RETURN_IF_ERROR(Path::Parse(
      external_data->GetRelPath(), external_data_relative_path));

  std::vector<char> buffer(actual_tensor_data_length);

  ORT_RETURN_IF_ERROR(Env::Default().ReadFileIntoBuffer(
      (model_path.ParentPath() / external_data_relative_path).ToPathString().c_str(),
      external_data->GetOffset(),
      actual_tensor_data_length,
      gsl::make_span(buffer)));

  raw_data = std::move(buffer);

  return Status::OK();
}
}  // namespace onnxruntime

#endif  // !(ORT_MINIMAL_BUILD)
