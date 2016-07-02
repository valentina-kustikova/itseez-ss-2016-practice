#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include "workaround.hpp"

using namespace std;
using namespace cv;

const char* kAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* kOptions =
    "{ @image         |        | image to process         }"
    "{ h ? help usage |        | print help message       }";


int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, kOptions);
  parser.about(kAbout);

  // If help option is given, print help message and exit.
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  // Read image.
  Mat src = imread(parser.get<string>(0), CV_LOAD_IMAGE_GRAYSCALE);
  if (src.empty()) {
    cout << "Failed to open image file '" + parser.get<string>(0) + "'." 
      << endl;
    return 0;
  }

  // Show source image.
  const string kSrcWindowName = "Source image";
  const int kWaitKeyDelay = 1;  
  namedWindow(kSrcWindowName);
  imshow(kSrcWindowName, src);
  waitKey(kWaitKeyDelay);

  // Threshold data.
  MatrixProcessorImpl processor;
  const int threshold = 128;
  processor.Threshold(src.data, src.cols, src.rows, threshold);

  // Show destination image.
  const string kDstWindowName = "Destination image";
  namedWindow(kDstWindowName);
  imshow(kDstWindowName, src);
  waitKey();

  return 0;
}
