#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

// tensorrt
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

// opencv
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda_utils.h>
#include <mish_plugin.hpp>
#include <nms_plugin.hpp>
#include <yolo_layer_plugin.hpp>

struct deleter
{
  template <typename T>
  void operator()(T * obj) const
  {
    if (obj) {
      obj->destroy();
    }
  }
};

class Logger : public nvinfer1::ILogger
{
public:
  Logger(bool verbose) : _verbose(verbose) {}

  void log(Severity severity, const char * msg) override
  {
    if (_verbose || (severity != Severity::kINFO) && (severity != Severity::kVERBOSE))
      std::cout << msg << std::endl;
  }

private:
  bool _verbose{false};
};

std::vector<float> prepareImage(const cv::Mat & img, const int c, const int w, const int h)
{
  using namespace cv;

  float scale = min(float(w) / img.cols, float(h) / img.rows);
  auto scaleSize = cv::Size(img.cols * scale, img.rows * scale);

  cv::Mat rgb;
  cv::cvtColor(img, rgb, CV_BGR2RGB);
  cv::Mat resized;
  cv::resize(rgb, resized, scaleSize, 0, 0, INTER_CUBIC);

  cv::Mat cropped(h, w, CV_8UC3, 127);
  Rect rect(
    (w - scaleSize.width) / 2, (h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);
  resized.copyTo(cropped(rect));

  cv::Mat img_float;
  if (c == 3)
    cropped.convertTo(img_float, CV_32FC3, 1 / 255.0);
  else
    cropped.convertTo(img_float, CV_32FC1, 1 / 255.0);

  //HWC TO CHW
  std::vector<Mat> input_channels(c);
  cv::split(img_float, input_channels);

  std::vector<float> result(h * w * c);
  auto data = result.data();
  int channelLength = h * w;
  for (int i = 0; i < c; ++i) {
    memcpy(data, input_channels[i].data, channelLength * sizeof(float));
    data += channelLength;
  }

  return result;
}

int main(int argc, char * argv[])
{
  assert(argc == 4);
  Logger logger(true);

  std::chrono::high_resolution_clock::time_point start, end;

  auto builder = std::unique_ptr<nvinfer1::IBuilder, deleter>(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    return false;
  }
  const auto explicitBatch =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network =
    std::unique_ptr<nvinfer1::INetworkDefinition, deleter>(builder->createNetworkV2(explicitBatch));
  if (!network) {
    return false;
  }
  auto config = std::unique_ptr<nvinfer1::IBuilderConfig, deleter>(builder->createBuilderConfig());
  if (!config) {
    return false;
  }
  auto parser =
    std::unique_ptr<nvonnxparser::IParser, deleter>(nvonnxparser::createParser(*network, logger));
  if (!parser) {
    return false;
  }
  if (!parser->parseFromFile(argv[1], static_cast<int>(nvinfer1::ILogger::Severity::kERROR))) {
    return false;
  }
  std::vector<nvinfer1::ITensor *> scores, boxes, classes;
  auto input = network->getInput(0);
  auto num_outputs = network->getNbOutputs();
  // float anchors[3][6] = {116, 90, 156, 198, 373, 326, 30, 61, 62,
  //                        45,  59, 119, 10,  13,  16,  30, 33, 23};
  float anchors[3][6] = {12, 16, 19,  36,  40,  28,  36,  75,  76,
                         55, 72, 146, 142, 110, 192, 243, 459, 401};
  // float anchors[3][6] = {10, 13, 16,  30,  33, 23,  30,  61,  62,
  //                        45, 59, 119, 116, 90, 156, 198, 373, 326};
  float scale_x_y[3] = {1.2, 1.1, 1.05};
  // float scale_x_y[3] = {1.0, 1.0, 1.0};
  auto input_dims = input->getDimensions();
  auto input_channel = input_dims.d[1];
  auto input_width = input_dims.d[2];
  auto input_height = input_dims.d[3];
  for (int i = 0; i < num_outputs; ++i) {
    auto output = network->getOutput(i);
    std::vector<float> anchor(std::begin(anchors[i]), std::end(anchors[i]));
    auto yoloLayerPlugin =
      yolo::YoloLayerPlugin(input_width, input_height, 3, anchor, scale_x_y[i], 0.1);
    std::vector<nvinfer1::ITensor *> inputs = {output};
    auto layer = network->addPluginV2(inputs.data(), inputs.size(), yoloLayerPlugin);
    scores.push_back(layer->getOutput(0));
    boxes.push_back(layer->getOutput(1));
    classes.push_back(layer->getOutput(2));
  }

  // Cleanup outputs
  for (int i = 0; i < num_outputs; i++) {
    auto output = network->getOutput(0);
    network->unmarkOutput(*output);
  }

  // Concat tensors from each feature map
  std::vector<nvinfer1::ITensor *> concat;
  for (auto tensors : {scores, boxes, classes}) {
    auto layer = network->addConcatenation(tensors.data(), tensors.size());
    layer->setAxis(1);
    auto output = layer->getOutput(0);
    concat.push_back(output);
  }

  // Add NMS plugin
  auto nmsPlugin = yolo::NMSPlugin(0.2, 100);
  auto layer = network->addPluginV2(concat.data(), concat.size(), nmsPlugin);
  for (int i = 0; i < layer->getNbOutputs(); i++) {
    auto output = layer->getOutput(i);
    network->markOutput(*output);
  }

  int max_batch_size = 1;
  builder->setMaxBatchSize(max_batch_size);
  config->setMaxWorkspaceSize(1 << 30);
  // config->setFlag(nvinfer1::BuilderFlag::kFP16);
  std::unique_ptr<nvinfer1::ICudaEngine, deleter> engine;
  std::ifstream file(argv[2], std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    std::cout << "Create engine..." << std::endl;
    engine = std::unique_ptr<nvinfer1::ICudaEngine, deleter>(
      builder->buildEngineWithConfig(*network, *config));
    if (!engine) {
      std::cout << "Fail to create engine" << std::endl;
      return false;
    }
    std::cout << "Successfully create engine" << std::endl;
    auto selialized = std::unique_ptr<nvinfer1::IHostMemory, deleter>(engine->serialize());
    std::cout << "Save engine: " << argv[2] << std::endl;
    std::ofstream file;
    file.open(argv[2], std::ios::binary | std::ios::out);
    if (!file.is_open()) return false;
    file.write((const char *)selialized->data(), selialized->size());
    file.close();
  } else {
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    char * buffer = new char[size];
    file.read(buffer, size);
    file.close();
    auto runtime =
      std::unique_ptr<nvinfer1::IRuntime, deleter>(nvinfer1::createInferRuntime(logger));
    engine = std::unique_ptr<nvinfer1::ICudaEngine, deleter>(
      runtime->deserializeCudaEngine(buffer, size, nullptr));
    delete[] buffer;
  }

  std::cout << "Preparing data..." << std::endl;
  start = std::chrono::high_resolution_clock::now();
  auto orig_image = cv::imread(argv[3], cv::IMREAD_COLOR);
  auto image = orig_image.clone();
  cv::resize(image, image, cv::Size(input_width, input_height));
  cv::Mat pixels;
  image.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);
  std::vector<float> data(input_channel * input_width * input_height);
  data = prepareImage(image, input_channel, input_width, input_height);

  // inference

  auto context =
    std::unique_ptr<nvinfer1::IExecutionContext, deleter>(engine->createExecutionContext());
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  auto data_d = cuda::make_unique<float[]>(input_channel * input_width * input_height);
  cudaMemcpy(data_d.get(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
  auto out_scores_d = cuda::make_unique<float[]>(100);
  auto out_boxes_d = cuda::make_unique<float[]>(4 * 100);
  auto out_classes_d = cuda::make_unique<float[]>(100);
  std::vector<void *> buffers = {data_d.get(), out_scores_d.get(), out_boxes_d.get(),
                                 out_classes_d.get()};
  context->enqueueV2(buffers.data(), stream, nullptr);

  auto out_scores = std::make_unique<float[]>(100);
  auto out_boxes = std::make_unique<float[]>(4 * 100);
  auto out_classes = std::make_unique<float[]>(100);
  cudaMemcpyAsync(
    out_scores.get(), out_scores_d.get(), sizeof(float) * 100, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(
    out_boxes.get(), out_boxes_d.get(), sizeof(float) * 4 * 100, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(
    out_classes.get(), out_classes_d.get(), sizeof(float) * 100, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  auto out_image = orig_image.clone();
  for (int i = 0; i < 100; ++i) {
    if (out_scores[i] < 0.5) break;
    std::cout << out_classes[i] << std::endl;
    const auto x = out_boxes[i * 4] * orig_image.cols;
    const auto y = out_boxes[i * 4 + 1] * orig_image.rows;
    const auto w = out_boxes[i * 4 + 2] * orig_image.cols;
    const auto h = out_boxes[i * 4 + 3] * orig_image.rows;
    cv::Point2f tl(x, y);
    cv::Point2f br(x + w, y + h);
    cv::rectangle(out_image, tl, br, cv::Scalar(255, 255 / 80 * out_scores[i], 0), 3);
  }
  end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "exec time: " << elapsed << std::endl;
  cv::imwrite("output.jpg", out_image);
  return 0;
}
