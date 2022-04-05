// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #if defined(USE_CUDA) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "contrib_ops/cpu/bert/longformer_attention_base.h"

#include <numeric>
#include <functional>
#include <iostream>

namespace onnxruntime {
namespace test {

enum OrderCublasLt {
  ORDER_COL = 0,
  ORDER_ROW = 1,
  ORDER_COL32 = 2,
  ORDER_COL4_4R2_8C = 3,
  ORDER_COL32_2R_4R4 = 4
};

template <typename T>
static std::vector<T> GenData(std::vector<int64_t> const& shape, float scale) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  scale = std::is_same<T, int8_t>::value ? 1.0f : scale;  // using scale = 1.0f to generate int8_t data,
  std::vector<T> r(n);
  RandomValueGenerator random{};
  std::vector<int> tmp = random.Uniform<int32_t>(shape, -128, 127);
  for (int64_t i = 0; i < n; i++) {
    r[i] = static_cast<T>(tmp[i] * scale);
  }
  return r;
}

class OrderedIndex {
  OrderCublasLt order_;
  int64_t rows_;
  int64_t cols_;

 public:
  OrderedIndex(OrderCublasLt order, int64_t rows, int64_t cols)
      : order_(order), rows_(rows), cols_(cols) {}

  int64_t operator()(int64_t r, int64_t c);
};

int64_t OrderedIndex::operator()(int64_t r, int64_t c) {
  switch (order_) {
    case ORDER_ROW:
      return r * cols_ + c;
    case ORDER_COL:
      return c * rows_ + r;
    case ORDER_COL32: {
      int64_t tile_id = c / 32;
      int64_t tile_stride = 32 * rows_;
      return tile_id * tile_stride + r * 32 + (c % 32);
    }
    case ORDER_COL4_4R2_8C: {
      int64_t tiles_c = c / 32;
      int64_t tiles_r = r / 8;
      int64_t tile_idx = tiles_c * (rows_ / 8) + tiles_r;
      int64_t offset = tile_idx * (32 * 8);
      offset += (r & 0x1) * (32 * 4);
      int64_t in_4x4x8_tile_c = c % 32;
      int64_t in_4x4x8_tile_r = (r % 8) / 2;
      int64_t in_4x4x8_idx = (in_4x4x8_tile_c / 4) * (4 * 4) + in_4x4x8_tile_r * 4 + (in_4x4x8_tile_c % 4);
      offset += in_4x4x8_idx;
      return offset;
    }
    case ORDER_COL32_2R_4R4: {
      // TODO:
    }
    default:
      return 0;
  }
}

void BatchRowColFromShape(std::vector<int64_t> const& shape, int64_t& batch, int64_t& rows, int64_t& cols) {
  cols = shape.back();
  rows = (shape.size() > 1 ? shape[shape.size() - 2] : 1LL);
  batch = (shape.size() <= 2)
              ? 1LL
              : std::accumulate(shape.data(), shape.data() + (shape.size() - 2), 1LL, std::multiplies<int64_t>());
}

template <typename T>
static std::vector<int8_t> QuantizeTransform(std::vector<int64_t> const& shape, float scale,
                                             const std::vector<T>& src, OrderCublasLt order) {
  int64_t cols = 0, rows = 0, batch = 0;
  BatchRowColFromShape(shape, batch, rows, cols);

  OrderedIndex src_indexer(ORDER_ROW, rows, cols);
  OrderedIndex dst_indexer(order, rows, cols);

  std::vector<int8_t> dst(batch * cols * rows, 0);
  const T* bsrc = src.data();
  int8_t* bdst = dst.data();
  for (int64_t b = 0, batch_stride = rows * cols; b < batch; b++) {
    for (int64_t r = 0; r < rows; r++) {
      for (int64_t c = 0; c < cols; c++) {
        int64_t src_idx = src_indexer(r, c);
        int64_t dst_idx = dst_indexer(r, c);
        if (src_idx >= batch_stride || dst_idx >= batch_stride || bdst[dst_idx] != 0) {
          std::cout << "out of bound index calculated, error found in OrderedIndexer" << std::endl;
        }
        float v = (float)bsrc[src_idx] / scale;
        v = std::max(-128.0f, v);
        v = std::min(127.0f, v);
        bdst[dst_idx] = static_cast<int8_t>(std::round(v));
      }
    }
    bsrc += batch_stride;
    bdst += batch_stride;
  }
  return dst;
}

template <typename T>
static std::vector<T> DequantizeTransform(std::vector<int64_t> const& shape, float scale,
                                          const std::vector<int8_t>& src, OrderCublasLt order) {
  int64_t cols = 0, rows = 0, batch = 0;
  BatchRowColFromShape(shape, batch, rows, cols);

  OrderedIndex src_indexer(order, rows, cols);
  OrderedIndex dst_indexer(ORDER_ROW, rows, cols);

  std::vector<T> dst(batch * cols * rows, T(0.0f));
  const int8_t* bsrc = src.data();
  T* bdst = dst.data();
  for (int64_t b = 0, batch_stride = rows * cols; b < batch; b++) {
    for (int64_t r = 0; r < rows; r++) {
      for (int64_t c = 0; c < cols; c++) {
        int64_t src_idx = src_indexer(r, c);
        int64_t dst_idx = dst_indexer(r, c);
        if (src_idx >= batch_stride || dst_idx >= batch_stride || bdst[dst_idx] != T(0.0f)) {
          std::cout << "out of bound index calculated, error found in OrderedIndexer" << std::endl;
        }
        T v = T(float(scale) * float(bsrc[src_idx]));
        bdst[dst_idx] = v;
      }
    }
    bsrc += batch_stride;
    bdst += batch_stride;
  }
  return dst;
}

template <typename T>
static void RunQOrdered_Quantize_Test(
    std::vector<T> const& fvec,
    std::vector<int64_t> const& shape,
    OrderCublasLt order_q,
    T scale) {
  auto qvec = QuantizeTransform(shape, scale, fvec, order_q);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  OpTester test_q("QuantizeWithOrder", 1, onnxruntime::kMSDomain);
  test_q.AddAttribute("order_input", (int64_t)ORDER_ROW);
  test_q.AddAttribute("order_output", (int64_t)order_q);
  test_q.AddInput<T>("input", shape, fvec);
  test_q.AddInput<T>("scale_input", {}, {scale});
  test_q.AddOutput("output", shape, qvec);
  test_q.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QOrderedTest, FP32_Quantize_COL32) {
  std::vector<int64_t> shape = {1, 5, 32 * 2};
  float scale = 1.0f;
  std::vector<float> fvec = GenData<float>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL32, scale);
}

TEST(QOrderedTest, FP16_Quantize_COL32) {
  std::vector<int64_t> shape = {1, 5, 32 * 2};
  MLFloat16 scale = MLFloat16(1.0f);
  std::vector<MLFloat16> fvec = GenData<MLFloat16>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL32, scale);
}

TEST(QOrderedTest, FP32_Quantize_COL4_4R2_8C) {
  std::vector<int64_t> shape = {1, 8 * 3, 32 * 2};
  float scale(1.0f);
  std::vector<float> fvec = GenData<float>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL4_4R2_8C, scale);
}

template <typename T>
static void RunQOrdered_Dequantize_Test(
    std::vector<int8_t> const& qvec,
    std::vector<int64_t> const& shape,
    OrderCublasLt order_q,
    T scale) {
  auto fvec = DequantizeTransform<T>(shape, scale, qvec, order_q);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  OpTester test_dq("DequantizeWithOrder", 1, onnxruntime::kMSDomain);
  test_dq.AddAttribute("order_input", (int64_t)order_q);
  test_dq.AddAttribute("order_output", (int64_t)ORDER_ROW);
  test_dq.template AddInput("input", shape, qvec);
  test_dq.AddInput<T>("scale_input", {}, {scale});
  test_dq.AddOutput("output", shape, fvec);
  test_dq.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Dequantize only work for ORDER_COL32 input
TEST(QOrderedTest, FP32_Dequantize_COL32) {
  std::vector<int64_t> shape = {1, 5, 32 * 2};
  float scale = 1.0f;
  std::vector<int8_t> qvec = GenData<int8_t>(shape, 1.0f);
  RunQOrdered_Dequantize_Test(qvec, shape, ORDER_COL32, scale);
}

// Dequantize only work for ORDER_COL32 input
TEST(QOrderedTest, FP16_Dequantize_COL32) {
  std::vector<int64_t> shape = {1, 5, 32 * 2};
  MLFloat16 scale(1.0f);
  std::vector<int8_t> qvec = GenData<int8_t>(shape, 1.0f);
  RunQOrdered_Dequantize_Test(qvec, shape, ORDER_COL32, scale);
}

static void RunQOrdered_MatMul_Test(
    std::vector<int64_t> const& shapeA,
    std::vector<int64_t> const& shapeB,
    std::vector<int64_t> const& shapeY,
    OrderCublasLt order_weight,
    float scaleA, float scaleB, float scaleC, float scaleY,
    bool add_bias = false, bool broadcast_c_batch = false) {
  int64_t nY = std::accumulate(shapeY.begin(), shapeY.end(), int64_t{1LL}, std::multiplies<int64_t>());
  std::vector<int8_t> vecA = GenData<int8_t>(shapeA, 1.0f);
  std::vector<int8_t> vecB = GenData<int8_t>(shapeB, 1.0f);
  std::vector<int8_t> vecY(nY);

  int64_t colsA = 0, rowsA = 0, batchA = 0;
  BatchRowColFromShape(shapeA, batchA, rowsA, colsA);
  OrderedIndex indexerA(ORDER_COL32, rowsA, colsA);

  int64_t colsB = 0, rowsB = 0, batchB = 0;
  BatchRowColFromShape(shapeB, batchB, rowsB, colsB);
  OrderedIndex indexerB(order_weight, colsB, rowsB);  // B need Transpose

  int64_t colsY = 0, rowsY = 0, batchY = 0;
  BatchRowColFromShape(shapeY, batchY, rowsY, colsY);
  OrderedIndex indexerY(ORDER_COL32, rowsY, colsY);

  std::vector<int64_t> shapeBias = {colsY};
  std::vector<float> vecBias;
  if (add_bias) {
    vecBias = GenData<float>(shapeBias, scaleY);
  }
  std::vector<int8_t> vecC;
  std::vector<int64_t> shapeC = {broadcast_c_batch ? 1 : batchY, rowsY, colsY};
  if (scaleC != 0.0f) {
    vecC = GenData<int8_t>(shapeC, 1.0f);
  }

  // TODO: brocasting higher dims
  float alpha = scaleA * scaleB / scaleY;
  int8_t const* A = vecA.data();
  int8_t const* B = vecB.data();
  int8_t* Y = vecY.data();
  int8_t* C = vecC.data();
  for (int64_t b = 0; b < batchY; b++) {
    for (int64_t m = 0; m < rowsA; m++) {
      for (int64_t n = 0; n < colsB; n++) {
        float sum = 0.0f;
        for (int64_t k = 0; k < colsA; k++) {
          auto posA = indexerA(m, k);
          auto posB = indexerB(n, k);  // Transpose B
          sum += A[posA] * B[posB];
        }
        sum *= alpha;
        if (add_bias) {
          sum += vecBias[n];
        }
        auto posY = indexerY(m, n);
        if (scaleC != 0.0f) {
          sum += scaleC * C[posY];
        }
        Y[posY] = static_cast<int8_t>((int)std::round(std::min(127.0f, std::max(-128.0f, sum))));
      }
    }
    A += (batchA <= 1 ? int64_t{0} : (rowsA * colsA));
    B += (batchB <= 1 ? int64_t{0} : (rowsB * colsB));
    Y += (batchY <= 1 ? int64_t{0} : (rowsY * colsY));
    C += (shapeC[0] == 1 ? int64_t{0} : (rowsY * colsY));
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  OpTester test_dq("QOrderedMatMul", 1, onnxruntime::kMSDomain);
  test_dq.AddAttribute("order_A", (int64_t)ORDER_COL32);
  test_dq.AddAttribute("order_B", (int64_t)order_weight);
  test_dq.AddAttribute("order_Y", (int64_t)ORDER_COL32);
  test_dq.AddInput<int8_t>("A", shapeA, vecA);
  test_dq.AddInput<float>("scale_A", {}, {scaleA});
  test_dq.AddInput<int8_t>("B", shapeB, vecB);
  test_dq.AddInput<float>("scale_B", {}, {scaleB});
  test_dq.AddInput<float>("scale_Y", {}, {scaleY});
  if (add_bias) {
    test_dq.AddInput<float>("bias", shapeBias, vecBias);
  }
  if (scaleC != 0.0f) {
    test_dq.AddInput<int8_t>("C", shapeC, vecC);
    test_dq.AddInput<float>("scale_C", {}, {scaleC});
  }
  test_dq.AddOutput<int8_t>("Y", shapeY, vecY, false, 0.0f, 1.0f /* abs error */);
  test_dq.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QOrderedTest, MatMul_COL4_4R2_8C_16x32x32) {
  std::vector<int64_t> shapeA = {16, 32};
  std::vector<int64_t> shapeB = {32, 32};
  std::vector<int64_t> shapeY = {16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_bias_COL4_4R2_8C_16x32x32) {
  std::vector<int64_t> shapeA = {16, 32};
  std::vector<int64_t> shapeB = {32, 32};
  std::vector<int64_t> shapeY = {16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_COL4_4R2_8C_16x32x32) {
  std::vector<int64_t> shapeA = {16, 32};
  std::vector<int64_t> shapeB = {32, 32};
  std::vector<int64_t> shapeY = {16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_bias_addC_COL4_4R2_8C_16x32x32) {
  std::vector<int64_t> shapeA = {16, 32};
  std::vector<int64_t> shapeB = {32, 32};
  std::vector<int64_t> shapeY = {16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, true /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_COL4_4R2_8C_16x32x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 32};
  std::vector<int64_t> shapeY = {2, 16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_bias_COL4_4R2_8C_16x32x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 32};
  std::vector<int64_t> shapeY = {2, 16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_COL4_4R2_8C_16x32x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 32};
  std::vector<int64_t> shapeY = {2, 16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_broadcastC_COL4_4R2_8C_16x32x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 32};
  std::vector<int64_t> shapeY = {2, 16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, true /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_bias_COL4_4R2_8C_16x32x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 32};
  std::vector<int64_t> shapeY = {2, 16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_bias_addC_broadcastC_COL4_4R2_8C_16x32x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 32};
  std::vector<int64_t> shapeY = {2, 16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, true /* broadcast batch c */);
}

}  // namespace test
}  // namespace onnxruntime

// #endif
