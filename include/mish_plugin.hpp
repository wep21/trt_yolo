#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "NvInferPlugin.h"

#define CHECK(status)                                           \
    do {                                                        \
        auto ret = status;                                      \
        if (ret != 0) {                                         \
            std::cerr << "Cuda failure in file '" << __FILE__   \
                      << "' line " << __LINE__                  \
                      << ": " << ret << std::endl;              \
            abort();                                            \
        }                                                       \
    } while (0)

namespace yolo
{
class MishPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
  explicit MishPlugin();
  MishPlugin(const void * data, size_t length);

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
  int mish(cudaStream_t stream, const float * input, float * output, const int n);

private:
  const char * mPluginNamespace;

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

class MishPluginCreator : public nvinfer1::IPluginCreator
{
public:
  MishPluginCreator();

  ~MishPluginCreator() override = default;

  const char * getPluginName() const override;

  const char * getPluginVersion() const override;

  const nvinfer1::PluginFieldCollection * getFieldNames() override;

  nvinfer1::IPluginV2DynamicExt * createPlugin(
    const char * name, const nvinfer1::PluginFieldCollection * fc) override;

  nvinfer1::IPluginV2DynamicExt * deserializePlugin(
    const char * name, const void * serialData, size_t serialLength) override;

  void setPluginNamespace(const char * libNamespace) override { mNamespace = libNamespace; }

  const char * getPluginNamespace() const override { return mNamespace.c_str(); }

private:
  std::string mNamespace;
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(MishPluginCreator);

}  // namespace yolo
