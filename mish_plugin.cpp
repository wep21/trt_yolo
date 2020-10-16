#include "mish_plugin.hpp"
#include <stdio.h>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace nvinfer1;

namespace
{
const char * MISH_PLUGIN_VERSION{"1"};
const char * MISH_PLUGIN_NAME{"Mish_TRT"};

inline int64_t volume(const nvinfer1::Dims & d)
{
  int64_t v = 1;
  for (int64_t i = 0; i < d.nbDims; i++) v *= d.d[i];
  return v;
}
}  // namespace

namespace yolo
{
MishPlugin::MishPlugin() {}

// create the plugin at runtime from a byte stream
MishPlugin::MishPlugin(const void * data, size_t length) {}

// IPluginV2 Methods

const char * MishPlugin::getPluginType() const { return MISH_PLUGIN_NAME; }

const char * MishPlugin::getPluginVersion() const { return MISH_PLUGIN_VERSION; }

int MishPlugin::getNbOutputs() const { return 1; }

int MishPlugin::initialize() { return 0; }

void MishPlugin::terminate() {}

size_t MishPlugin::getSerializationSize() const { return 0; }

void MishPlugin::serialize(void * buffer) const {}

void MishPlugin::destroy() { delete this; }

void MishPlugin::setPluginNamespace(const char * pluginNamespace)
{
  mPluginNamespace = pluginNamespace;
}

const char * MishPlugin::getPluginNamespace() const { return mPluginNamespace; }

// IPluginV2Ext Methods

DataType MishPlugin::getOutputDataType(int index, const DataType * inputTypes, int nbInputs) const
{
  assert(inputTypes[0] == DataType::kFLOAT);
  return inputTypes[0];
}

// IPluginV2DynamicExt Methods

IPluginV2DynamicExt * MishPlugin::clone() const
{
  auto plugin = new MishPlugin(*this);
  plugin->setPluginNamespace(mPluginNamespace);
  return plugin;
}

DimsExprs MishPlugin::getOutputDimensions(
  int outputIndex, const DimsExprs * inputs, int nbInputs, IExprBuilder & exprBuilder)
{
  return inputs[0];
}

bool MishPlugin::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc * inOut, int nbInputs, int nbOutputs)
{
  return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == PluginFormat::kLINEAR;
}

void MishPlugin::configurePlugin(
  const DynamicPluginTensorDesc * in, int nbInput, const DynamicPluginTensorDesc * out,
  int nbOutput)
{
  assert(nbInput == 1);
  assert(nbOutput == 1);
}

size_t MishPlugin::getWorkspaceSize(
  const nvinfer1::PluginTensorDesc * inputs, int nbInputs,
  const nvinfer1::PluginTensorDesc * outputs, int nbOutputs) const
{
  return 0;
}

int MishPlugin::enqueue(
  const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
  const void * const * inputs, void * const * outputs, void * workspace, cudaStream_t stream)
{
  const int input_volume = volume(inputDesc[0].dims);

  int status = -1;

  const float * input = static_cast<const float *>(inputs[0]);
  float * output = static_cast<float *>(outputs[0]);
  status = mish(stream, input, output, input_volume);
  return status;
}

PluginFieldCollection MishPluginCreator::mFC{};
std::vector<PluginField> MishPluginCreator::mPluginAttributes;

MishPluginCreator::MishPluginCreator()
{
  mPluginAttributes.clear();

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char * MishPluginCreator::getPluginName() const { return MISH_PLUGIN_NAME; }

const char * MishPluginCreator::getPluginVersion() const { return MISH_PLUGIN_VERSION; }

const PluginFieldCollection * MishPluginCreator::getFieldNames() { return &mFC; }

IPluginV2DynamicExt * MishPluginCreator::createPlugin(const char * name, const PluginFieldCollection * fc)
{
  MishPlugin * obj = new MishPlugin();
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

IPluginV2DynamicExt * MishPluginCreator::deserializePlugin(
  const char * name, const void * serialData, size_t serialLength)
{
  // This object will be deleted when the network is destroyed, which will
  // call MishPlugin::destroy()
  MishPlugin * obj = new MishPlugin(serialData, serialLength);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}
}  // namespace yolo
