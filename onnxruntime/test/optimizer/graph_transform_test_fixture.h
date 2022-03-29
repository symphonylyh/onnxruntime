// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/framework/data_transfer_manager.h"
#include "core/framework/session_options.h"

#include "gtest/gtest.h"
#include "test/test_environment.h"

namespace onnxruntime {
namespace test {

class GraphTransformationTests : public ::testing::Test {
 protected:
  GraphTransformationTests();

  DataTransferManager dt_manager_;
  SessionOptions sess_options_;
  std::unique_ptr<logging::Logger> logger_;
};

}  // namespace test
}  // namespace onnxruntime
