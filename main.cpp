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

std::vector<float> prepareImage(const cv::Mat & img)
{
  using namespace cv;

  int c = 3;
  int h = 608;  //net h
  int w = 608;  //net w

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
  const int num_classes = 21;
  const int num_detection = 3000;
  const float score_threshold = 0.4;
  const float iou_threshold = 0.45;
  std::vector<std::string> label{
    "BG",        "aeroplane", "bicycle",     "bird",  "boat",        "bottle", "bus",
    "car",       "cat",       "chair",       "cow",   "diningtable", "dog",    "horse",
    "motorbike", "person",    "pottedplant", "sheep", "sofa",        "train",  "tvmonitor"};

  std::chrono::high_resolution_clock::time_point start, end;
  const int input_w = 608;
  const int input_h = 608;

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
  std::vector<nvinfer1::ITensor *> detections;
  auto input = network->getInput(0);
  auto num_outputs = network->getNbOutputs();
  float anchors[3][6] = {12, 16, 19,  36,  40,  28,  36,  75,  76,
                         55, 72, 146, 142, 110, 192, 243, 459, 401};
  float scale_x_y[3] = {1.2, 1.1, 1.05};
  for (int i = 0; i < num_outputs; ++i) {
    auto input_dims = input->getDimensions();
    auto input_width = input_dims.d[2];
    auto input_height = input_dims.d[3];
    auto output = network->getOutput(i);
    auto yoloLayerPlugin =
      yolo::YoloLayerPlugin(input_width, input_height, anchors[i], scale_x_y[i], 0.1);
    std::vector<nvinfer1::ITensor *> inputs = {output};
    auto layer = network->addPluginV2(inputs.data(), inputs.size(), yoloLayerPlugin);
    detections.push_back(layer->getOutput(0));
  }

  // Cleanup outputs
  for (int i = 0; i < num_outputs; i++) {
    auto output = network->getOutput(0);
    network->unmarkOutput(*output);
  }

  auto layer = network->addConcatenation(detections.data(), detections.size());
  auto concat_output = layer->getOutput(0);
  network->markOutput(*concat_output);

  int max_batch_size = 1;
  builder->setMaxBatchSize(max_batch_size);
  config->setMaxWorkspaceSize(1 << 24);
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
  cv::resize(image, image, cv::Size(input_w, input_h));
  cv::Mat pixels;
  image.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);
  const int channels = 3;
  std::vector<float> data(channels * input_w * input_h);
  data = prepareImage(image);

  // inference

  auto context =
    std::unique_ptr<nvinfer1::IExecutionContext, deleter>(engine->createExecutionContext());
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  auto data_d = cuda::make_unique<float[]>(channels * input_h * input_w);
  cudaMemcpy(data_d.get(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
  auto out_d = cuda::make_unique<float[]>(6 * (76 * 76 + 38 * 38 + 19 * 19) * 3);
  std::vector<void *> buffers = {data_d.get(), out_d.get()};
  context->enqueueV2(buffers.data(), stream, nullptr);

  auto out = std::make_unique<float[]>(6 * (76 * 76 + 38 * 38 + 19 * 19) * 3);
  cudaMemcpyAsync(
    out.get(), out_d.get(), sizeof(float) * 6 * (76 * 76 + 38 * 38 + 19 * 19) * 3,
    cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  /* auto out_image = orig_image.clone();
	for (int i = 0; i < num_classes; i++)
	{
		std::vector<float> probs;
		std::vector<cv::Rect2d> subset_boxes;
		std::vector<int> indices;
		for (int j = 0; j < num_detection; j++)
		{
			probs.push_back(scores[i + j * num_classes]);
			subset_boxes.push_back(cv::Rect2d(cv::Point2d(boxes[j * 4], boxes[j * 4 + 1]),
				                              cv::Point2d(boxes[j * 4 + 2], boxes[j * 4 + 3])));
			if (probs.size() == 0)
			{
				continue;
			}
		}
		cv::dnn::NMSBoxes(subset_boxes, probs, score_threshold, iou_threshold, indices);
		for (const auto& index: indices)
		{
			if (i != 0)
			{
				cv::Point2f tl(subset_boxes[index].tl().x * orig_image.cols, subset_boxes[index].tl().y * orig_image.rows);
				cv::Point2f br(subset_boxes[index].br().x * orig_image.cols, subset_boxes[index].br().y * orig_image.rows);
				cv::rectangle(out_image, tl, br, cv::Scalar(255, 255 / num_classes * i, 0), 3);
				cv::putText(out_image, label[i] + ": " + std::to_string(probs[index]), cvPoint(tl.x, tl.y -10), CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255 / num_classes * i, 0), 3);
			}
		}
	}*/
  end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "exec time: " << elapsed << std::endl;
  // cv::imwrite("output.jpg", out_image);
  return 0;
}
