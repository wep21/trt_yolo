#include "yolo_layer_plugin.hpp"
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>
#include <cassert>
#include <cmath>

using namespace nvinfer1;

namespace
{
const char * YOLO_LAYER_PLUGIN_VERSION{"1"};
const char * YOLO_LAYER_PLUGIN_NAME{"Yolo_Layer_TRT"};
const char * YOLO_LAYER_PLUGIN_NAMESPACE{""};

template <typename T>
void write(char *& buffer, const T & val)
{
  *reinterpret_cast<T *>(buffer) = val;
  buffer += sizeof(T);
}

template <typename T>
void read(const char *& buffer, T & val)
{
  val = *reinterpret_cast<const T *>(buffer);
  buffer += sizeof(T);
}
}  // namespace

namespace yolo
{
YoloLayerPlugin::YoloLayerPlugin(
  int width, int height, int num_anchors, std::vector<float> & anchors, float scale_xy,
  float score_thresh)
: width_(width),
  height_(height),
  num_anchors_(num_anchors),
  anchors_(anchors),
  scale_xy_(scale_xy),
  score_thresh_(score_thresh)
{
}

// create the plugin at runtime from a byte stream
YoloLayerPlugin::YoloLayerPlugin(const void * data, size_t length)
{
  const char *d = static_cast<const char *>(data), *a = d;
  read(d, width_);
  read(d, height_);
  read(d, num_anchors_);
  int anchor_size = num_anchors_ * 2;
  while (--anchor_size) {
    float val;
    read(d, val);
    anchors_.push_back(val);
  }
  read(d, scale_xy_);
  read(d, score_thresh_);

  assert(d == a + length);
}

// IPluginV2 Methods

const char * YoloLayerPlugin::getPluginType() const { return YOLO_LAYER_PLUGIN_NAME; }

const char * YoloLayerPlugin::getPluginVersion() const { return YOLO_LAYER_PLUGIN_VERSION; }

int YoloLayerPlugin::getNbOutputs() const { return 1; }

int YoloLayerPlugin::initialize() { return 0; }

void YoloLayerPlugin::terminate() {}

size_t YoloLayerPlugin::getSerializationSize() const
{
  return sizeof(width_) + sizeof(height_) + sizeof(num_anchors_) +
         sizeof(float) * num_anchors_ * 2 + sizeof(scale_xy_) + sizeof(score_thresh_);
}

void YoloLayerPlugin::serialize(void * buffer) const
{
  char *d = reinterpret_cast<char *>(buffer), *a = d;
  write(d, width_);
  write(d, height_);
  write(d, num_anchors_);
  for (int i = 0; i < num_anchors_ * 2; ++i) {
    write(d, anchors_[i]);
  }
  write(d, scale_xy_);
  write(d, score_thresh_);

  assert(d == a + getSerializationSize());
}

void YoloLayerPlugin::destroy() { delete this; }

void YoloLayerPlugin::setPluginNamespace(const char * pluginNamespace)
{
  mPluginNamespace = pluginNamespace;
}

const char * YoloLayerPlugin::getPluginNamespace() const { return mPluginNamespace; }

// IPluginV2Ext Methods

DataType YoloLayerPlugin::getOutputDataType(
  int index, const DataType * inputTypes, int nbInputs) const
{
  return DataType::kFLOAT;
}

// IPluginV2DynamicExt Methods

IPluginV2DynamicExt * YoloLayerPlugin::clone() const
{
  auto plugin = new YoloLayerPlugin(*this);
  plugin->setPluginNamespace(mPluginNamespace);
  return plugin;
}

DimsExprs YoloLayerPlugin::getOutputDimensions(
  int outputIndex, const DimsExprs * inputs, int nbInputs, IExprBuilder & exprBuilder)
{
  DimsExprs ret = inputs[0];
  ret.nbDims = 3;
  const auto total_count =
    ret.d[2]->getConstantValue() * ret.d[3]->getConstantValue() * num_anchors_ * 6;
  ret.d[1] = exprBuilder.constant(total_count);
  ret.d[2] = exprBuilder.constant(1);
  ret.d[3] = exprBuilder.constant(1);
  return ret;
}

bool YoloLayerPlugin::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc * inOut, int nbInputs, int nbOutputs)
{
  return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == PluginFormat::kLINEAR;
}

void YoloLayerPlugin::configurePlugin(
  const DynamicPluginTensorDesc * in, int nbInput, const DynamicPluginTensorDesc * out,
  int nbOutput)
{
  assert(nbInput == 1);
  assert(nbOutput == 3);
}

size_t YoloLayerPlugin::getWorkspaceSize(
  const nvinfer1::PluginTensorDesc * inputs, int nbInputs,
  const nvinfer1::PluginTensorDesc * outputs, int nbOutputs) const
{
  return 0;
}

int YoloLayerPlugin::enqueue(
  const nvinfer1::PluginTensorDesc * inputDesc, const nvinfer1::PluginTensorDesc * outputDesc,
  const void * const * inputs, void * const * outputs, void * workspace, cudaStream_t stream)
{
  const int batch_size = inputDesc[0].dims.d[0];
  const int grid_width = inputDesc[0].dims.d[2];
  const int grid_height = inputDesc[0].dims.d[3];
  const int num_classes = inputDesc[0].dims.d[1] / num_anchors_ - 5;

  int status = -1;
  const float * input = static_cast<const float *>(inputs[0]);
  float * output = static_cast<float *>(outputs[0]);
  status = yoloLayer(
    inputDesc[0].dims.d[0], inputs, outputs, grid_width, grid_height, num_classes, num_anchors_,
    anchors_d_, width_, height_, scale_xy_, score_thresh_, stream);
  return status;
}

YoloLayerPluginCreator::YoloLayerPluginCreator() {}

const char * YoloLayerPluginCreator::getPluginName() const { return YOLO_LAYER_PLUGIN_NAME; }

const char * YoloLayerPluginCreator::getPluginVersion() const { return YOLO_LAYER_PLUGIN_VERSION; }

const char * YoloLayerPluginCreator::getPluginNamespace() const
{
  return YOLO_LAYER_PLUGIN_NAMESPACE;
}

void YoloLayerPluginCreator::setPluginNamespace(const char * N) {}
const PluginFieldCollection * YoloLayerPluginCreator::getFieldNames() { return nullptr; }

IPluginV2DynamicExt * YoloLayerPluginCreator::createPlugin(
  const char * name, const PluginFieldCollection * fc)
{
  return nullptr;
}

IPluginV2DynamicExt * YoloLayerPluginCreator::deserializePlugin(
  const char * name, const void * serialData, size_t serialLength)
{
  return new YoloLayerPlugin(serialData, serialLength);
}
}  // namespace yolo
