#include <stdio.h>
#include <stdexcept>
#include "math_constants.h"
#include "yolo_layer.h"

namespace
{
static constexpr size_t CUDA_ALIGN = 256;

template <typename T>
inline size_t get_size_aligned(size_t num_elem)
{
  size_t size = num_elem * sizeof(T);
  size_t extra_align = 0;
  if (size % CUDA_ALIGN != 0) {
    extra_align = CUDA_ALIGN - size % CUDA_ALIGN;
  }
  return size + extra_align;
}

template <typename T>
inline T * get_next_ptr(size_t num_elem, void *& workspace, size_t & workspace_size)
{
  size_t size = get_size_aligned<T>(num_elem);
  if (size > workspace_size) {
    throw std::runtime_error("Workspace is too small!");
  }
  workspace_size -= size;
  T * ptr = reinterpret_cast<T *>(workspace);
  workspace = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(workspace) + size);
  return ptr;
}
}  // namespace

namespace yolo
{
inline __device__ float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

inline __device__ float scaleSigmoid(float x, float scale)
{
  return scale * sigmoid(x) - (scale - 1.0f) * 0.5f;
}

template <unsigned TPB>
__global__ void yoloLayerKernel(
  const float * input, float * out_scores, float4 * out_boxes, float * out_classes, int grid_width,
  int grid_height, int num_classes, int num_anchors, const float * anchors, int input_width,
  int input_height, float scale_x_y, float score_thresh)
{
  int idx = threadIdx.x + TPB * blockIdx.x;
  int total_grids = grid_width * grid_height;
  if (idx >= total_grids * num_anchors) return;
  auto out_score = (out_scores) + idx;
  auto out_box = (out_boxes) + idx;
  auto out_class = (out_classes) + idx;

  int anchor_idx = idx / total_grids;
  idx = idx - total_grids * anchor_idx;
  int info_len = 5 + num_classes;
  auto cur_input = static_cast<const float *>(input) + anchor_idx * (info_len * total_grids);

  int class_id;
  float max_cls_logit = -CUDART_INF_F;  // minus infinity
  for (int i = 5; i < info_len; ++i) {
    float l = cur_input[idx + i * total_grids];
    if (l > max_cls_logit) {
      max_cls_logit = l;
      class_id = i - 5;
    }
  }
  float max_cls_prob = sigmoid(max_cls_logit);
  float objectness = sigmoid(cur_input[idx + 4 * total_grids]);

  int row = idx / grid_width;
  int col = idx % grid_width;

  auto x =
    (col + scaleSigmoid(cur_input[idx + 0 * total_grids], scale_x_y)) / grid_width;  // [0, 1]
  auto y =
    (row + scaleSigmoid(cur_input[idx + 1 * total_grids], scale_x_y)) / grid_height;  // [0, 1]
  auto w =
    __expf(cur_input[idx + 2 * total_grids]) * anchors[2 * anchor_idx] / input_width;  // [0, 1]
  auto h = __expf(cur_input[idx + 3 * total_grids]) * anchors[2 * anchor_idx + 1] /
           input_height;  // [0, 1]

  x -= w / 2;  // shift from center to top-left
  y -= h / 2;
  *out_box = make_float4(x, y, w, h);

  // det->det_confidence = box_prob;
  *out_class = class_id;
  *out_score = objectness < score_thresh ? 0.0 : max_cls_prob * objectness;
}

int yoloLayer(
  int batch_size, const void * const * inputs, void * const * outputs, int grid_width,
  int grid_height, int num_classes, int num_anchors, const std::vector<float> & anchors,
  int input_width, int input_height, float scale_x_y, float score_thresh, void * workspace,
  size_t workspace_size, cudaStream_t stream)
{
  if (!workspace || !workspace_size) {
    workspace_size = get_size_aligned<float>(anchors.size());
    return workspace_size;
  }

  auto anchors_d = get_next_ptr<float>(anchors.size(), workspace, workspace_size);
  cudaMemcpyAsync(
    anchors_d, anchors.data(), anchors.size() * sizeof *anchors_d, cudaMemcpyHostToDevice, stream);

  int num_elements = num_anchors * grid_width * grid_height;
  constexpr int block_size = 256;
  const int grid_size = (num_elements + block_size - 1) / block_size;
  for (int batch = 0; batch < batch_size; ++batch) {
    auto input = static_cast<const float *>(inputs[0]) +
                 batch * num_anchors * (num_classes + 5) * grid_width * grid_height;
    auto out_scores = static_cast<float *>(outputs[0]) + batch * num_elements;
    auto out_boxes = static_cast<float4 *>(outputs[1]) + batch * num_elements;
    auto out_classes = static_cast<float *>(outputs[2]) + batch * num_elements;
    yoloLayerKernel<block_size><<<grid_size, block_size, 0, stream>>>(
      input, out_scores, out_boxes, out_classes, grid_width, grid_height, num_classes, num_anchors,
      anchors_d, input_width, input_height, scale_x_y, score_thresh);
  }
  return 0;
}

}  // namespace yolo
