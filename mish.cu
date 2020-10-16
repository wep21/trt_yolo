#include <cuda_runtime_api.h>
#include <stdio.h>
#include "mish_plugin.hpp"

namespace yolo
{
__device__ float mish(float x)
{
  float e = __expf(x);
  float n = e * e + 2 * e;
  if (x <= -0.6f) return x * __fdividef(n, n + 2);

  return x - 2 * __fdividef(x, n + 2);
}
template <typename T, unsigned TPB>
__global__ void mishKernel(const T * input, T * output, int num_elem)
{
  int idx = threadIdx.x + TPB * blockIdx.x;
  if (idx >= num_elem) return;
  output[idx] = mish(input[idx]);
}

int MishPlugin::mish(cudaStream_t stream, const float * input, float * output, int n)
{
  constexpr int blockSize = 256;
  const int gridSize = (n + blockSize - 1) / blockSize;
  mishKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(input, output, n);

  CHECK(cudaPeekAtLastError());
  return 0;
}

}  // namespace yolo
