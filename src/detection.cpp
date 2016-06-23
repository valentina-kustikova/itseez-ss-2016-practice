#include "detection.hpp"

#include <algorithm>
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

using std::string;
using std::shared_ptr;
using std::vector;
using namespace cv;

shared_ptr<Detector> Detector::CreateDetector(const string& name) {
  if (name == "cascade") {
    return std::make_shared<CascadeDetector>();
  } else {
    std::cerr << "Failed to create detector with name '" << name << "'"
              << std::endl;
  }
  return nullptr;
}

bool CascadeDetector::Init(const string& model_file_path) {
  return detector.load(model_file_path);
}

void CascadeDetector::Detect(const Mat& frame, vector<Rect>& objects,
                             vector<double>& scores) {
  CV_Assert(!frame.empty());
  if (!detector.empty()) {
    vector<int> object_hits;
    const double kScaleFactor = 1.1;
    const int kMinHitsNum = 3;
    detector.detectMultiScale(frame, objects, object_hits, kScaleFactor,
                              kMinHitsNum);
    scores.resize(object_hits.size());
    std::copy(object_hits.begin(), object_hits.end(), scores.begin());
  } else {
    std::cerr << "Detector has not been initialized before usage." << std::endl;
  }
}
