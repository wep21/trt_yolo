#pragma once

#include <array>
#include <iostream>
#include <string>
#include <vector>
#include "NvInferPlugin.h"
#include "math_constants.h"

#define CHECK(status)                                                                           \
  do {                                                                                          \
    auto ret = status;                                                                          \
    if (ret != 0) {                                                                             \
      std::cerr << "Cuda failure in file '" << __FILE__ << "' line " << __LINE__ << ": " << ret \
                << std::endl;                                                                   \
      abort();                                                                                  \
    }                                                                                           \
  } while (0)

namespace yolo
{
static constexpr int NUM_ANCHORS = 3;
static constexpr float IGNORE_THRESH = 0.01f;

template <typename T>
struct alignas(T) Detection
{
  T bbox[4];  // x, y, w, h
  T confidence;
  T class_id;
};

class YoloLayerPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
  explicit YoloLayerPlugin(
    int width, int height, int num_classes, std::vector<float> & anchors, float scale_xy, float score_thresh);
  YoloLayerPlugin(const void * data, size_t length);

  // IPluginV2 methods
  const char * getPluginType() const override;
  const char * getPluginVersion() const override;
  int getNbOutputs() const override;
  int initialize() override;
  void terminate() override;
  size_t getSerializationSize() const override;
  void serialize(void * buffer) const override;
  void destroy() override;
  void setPluginNamespace(const char * libNamespace) override;
  const char * getPluginNamespace() const override;

  // IPluginV2Ext methods
  nvinfer1::DataType getOutputDataType(
    int index, const nvinfer1::DataType * inputType, int nbInputs) const override;

  // IPluginV2DynamicExt methods
  nvinfer1::IPluginV2DynamicExt * clone() const override;
  nvinfer1::DimsExprs getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs * inputs, int nbInputs,
    nvinfer1::IExprBuilder & exprBuilder) override;
  bool supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc * inOut, int nbInputs, int nbOutputs) override;
  void configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc * in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc * out, int nbOutputs) override;
  size_t getWorkspaceSize(
    const nvinfer1::PluginTensorDesc * inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc * outputs, int nbOutputs) const override;
  int enqueue(
    const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
    const void * const * inputs, void * const * outputs, void * workspace,
    cudaStream_t stream) override;
  int yoloLayer(
    int batch_size, const void * const * inputs, void * const * outputs, int grid_width, int grid_height,
    int num_classes, int num_anchors, const float * anchors_d, int input_width, int input_height,
    float scale_xy, float score_thresh, cudaStream_t stream);

private:
  const char * mPluginNamespace;
  int width_;
  int height_;
  int num_anchors_;
  float scale_xy_;
  std::vector<float> anchors_;
  float * anchors_d_;
  float score_thresh_;

protected:
  // To prevent compiler warnings.
  using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::configurePlugin;
  using nvinfer1::IPluginV2DynamicExt::enqueue;
  using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
  using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
  using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::supportsFormat;
};

class YoloLayerPluginCreator : public nvinfer1::IPluginCreator
{
public:
  YoloLayerPluginCreator();

  const char * getPluginName() const override;

  const char * getPluginVersion() const override;

  const nvinfer1::PluginFieldCollection * getFieldNames() override;

  nvinfer1::IPluginV2DynamicExt * createPlugin(
    const char * name, const nvinfer1::PluginFieldCollection * fc) override;

  nvinfer1::IPluginV2DynamicExt * deserializePlugin(
    const char * name, const void * serialData, size_t serialLength) override;

  void setPluginNamespace(const char * libNamespace) override;

  const char * getPluginNamespace() const override;
};

REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);

}  // namespace yolo
