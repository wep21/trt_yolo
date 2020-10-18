#pragma once
#include <cuda_runtime_api.h>
#include <vector>

namespace yolo
{
int yoloLayer(
  int batch_size, const void * const * inputs, void * const * outputs, int grid_width,
  int grid_height, int num_classes, int num_anchors, const std::vector<float> & anchors,
  int input_width, int input_height, float scale_xy, float score_thresh, void * workspace,
  size_t workspace_size, cudaStream_t stream);
}