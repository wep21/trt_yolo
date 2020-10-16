project(cublas_test)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 14)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

find_library(CUBLAS_LIB cublas HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(NVINFER NAMES nvinfer)
find_library(NVONNXPARSER nvonnxparser)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
  )

cuda_add_library(mish_plugin SHARED
  mish.cu
  mish_plugin.cpp
)

cuda_add_library(yolo_layer_plugin SHARED
  yolo_layer.cu
  yolo_layer_plugin.cpp
)

add_executable(trt_yolov4 main.cpp)
target_link_libraries(trt_yolov4
    ${OpenCV_LIBS}
    ${CUBLAS_LIB}
    ${NVINFER}
    ${NVONNXPARSER}
    ${NVINFER_PLUGIN}
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    mish_plugin
    yolo_layer_plugin
  )
