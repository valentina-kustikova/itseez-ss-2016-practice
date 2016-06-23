#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include "detection.hpp"
#include "tracking.hpp"

using namespace std;
using namespace cv;

const char* kAbout = "This app detects and tracks face in order to blur it.";

const char* kOptions =
    "{ v video        | <none> | video to process         }"
    "{ c camera       | <none> | camera to get video from }"
    "{ m model        | <none> |                          }"
    "{ h ? help usage |        | print help message       }";

int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, kOptions);
  parser.about(kAbout);

  // If help option is given, print help message and exit.
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }

  Mat frame;

  // Load input video.
  VideoCapture video;
  if (parser.has("video")) {
    string video_path = parser.get<String>("video");
    video.open(video_path);
    if (!video.isOpened()) {
      cout << "Failed to open video file '" << video_path << "'" << endl;
      return 0;
    }
  } else if (parser.has("camera")) {
    int camera_id = parser.get<int>("camera");
    video.open(camera_id);
    if (!video.isOpened()) {
      cout << "Failed to capture video stream from camera " << camera_id
           << endl;
      return 0;
    }
  }

  const string kWindowName = "video";
  const int kWaitKeyDelay = 10;
  const int kEscapeKey = 27;
  const Scalar kColorRed = CV_RGB(255, 0, 0);
  const Scalar kColorGreen = CV_RGB(0, 255, 0);
  const Scalar kColorBlue = CV_RGB(0, 0, 255);
  const Scalar kColorCyan = CV_RGB(0, 255, 255);

  namedWindow(kWindowName);

  CascadeDetector detector;
  string detector_model_file_path = parser.get<string>("model");
  if (!detector.Init(detector_model_file_path)) {
    cerr << "Failed to load detector from file '" << detector_model_file_path
         << "'" << endl;
    return 0;
  }

  MedianFlowTracker tracker;

  video >> frame;
  while (!frame.empty()) {
    Rect object;
    vector<Rect> objects;
    vector<double> scores;

    object = tracker.Track(frame);
    if (object == Rect()) {
      detector.Detect(frame, objects, scores);
      if (!objects.empty()) {
        auto max_score = std::max_element(scores.begin(), scores.end());
        size_t max_idx = std::distance(scores.begin(), max_score);
        object = objects[max_idx];
        tracker.Init(frame, object);
      }
    }

    if (object != Rect()) {
      int kernel_size = object.height / 6;
      kernel_size += 1 - kernel_size % 2;
      medianBlur(frame(object), frame(object), kernel_size);
    }

    imshow(kWindowName, frame);
    int key = waitKey(kWaitKeyDelay) & 0x00FF;
    if (key == kEscapeKey) {
      break;
    }
    video >> frame;
  }

  return 0;
}
