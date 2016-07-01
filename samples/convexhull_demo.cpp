#include <iostream>
#include <memory>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

const char* kWindowName = "Video";
const int kEscapeKey = 27;
const int kSpacebarKey = 32;
const int kSpeedLimit = 20;

const char* kOptions =
    "{ @image         |    | image to process             }"
    "{ d delay        | 25 | delay (in ms) between frames }"
    "{ h ? help usage |    | print help message           }";

const char* kAbout = "This is OpenCV sample application.";

struct EventData {
  Point point;
  vector<Point> points_position;
  vector<Point> points_direction;
};

int GetZVector(Point a, Point b, Point c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

void ConvexHull(vector<Point>& points, vector<size_t>& convex_hull) {
  const size_t size = points.size();
  if (points.empty()) {
    return;
  }

  convex_hull.resize(size);
  iota(convex_hull.begin(), convex_hull.end(), 0);

  auto min_max =
      minmax_element(points.begin(), points.begin() + size,
                     [](Point const& a, Point const& b) { return a.y < b.y; });
  Point& min = *min_max.first;

  iter_swap(convex_hull.begin() + (min_max.first - points.begin()),
            convex_hull.begin());

  auto end_iter = remove_if(convex_hull.begin() + 1, convex_hull.end(),
                            [&points, &min](const size_t id) {
                              Point& point = points[id];
                              return (point.x == min.x && point.y == min.y);
                            });

  convex_hull.erase(end_iter, convex_hull.end());

  if (convex_hull.size() >= 3) {
    sort(convex_hull.begin() + 1, convex_hull.end(),
         [&points, &min](const size_t& id1, const size_t& id2) -> bool {
           Point a = Point(points[id1].x - min.x, points[id1].y - min.y);
           Point b = Point(points[id2].x - min.x, points[id2].y - min.y);

           if (a.x * b.x <= 0) {
             return a.x <= b.x;
           } else {
             if (a.x < 0) {
               return a.x * b.y < a.y * b.x;
             } else {
               return a.y * b.x > a.x * b.y;
             }
           }
         });

    for (size_t i = 1; i < convex_hull.size() && convex_hull.size() > 2;) {
      size_t a = convex_hull[i - 1];
      size_t b = convex_hull[i];
      size_t c = convex_hull[(i + 1) % convex_hull.size()];
      if (GetZVector(points[a], points[b], points[c]) >= 0) {
        convex_hull.erase(convex_hull.begin() + (int)i);
        if (i != 1) {
          --i;
        }
      } else {
        i++;
      }
    }
  }
}

void Process(const Mat& in, Mat& out, const vector<Point>& points,
             vector<size_t>& convex_hull) {
  out = in.clone();

  const Scalar kColorRed = CV_RGB(255, 0, 0);
  const Scalar kColorGreen = CV_RGB(0, 255, 0);
  const Scalar kColorBlue = CV_RGB(0, 0, 255);

  for (size_t i = 0; i < points.size(); ++i) {
    circle(out, points[i], 1.5, kColorRed, 1);
    stringstream s;
    s << i;
    putText(out, s.str(), points[i], CV_FONT_HERSHEY_COMPLEX, 0.5, kColorBlue);
  }

  for (size_t i = 0; i < convex_hull.size(); ++i) {
    auto point_start = convex_hull[i];
    auto point_end = convex_hull[(i + 1) % convex_hull.size()];
    circle(out, points[point_start], 2, kColorRed, 2);
    line(out, points[point_start], points[point_end], kColorRed);
    stringstream s;
    s << i;
    putText(out, s.str(), points[point_start], CV_FONT_HERSHEY_COMPLEX, 1.,
            kColorGreen);
  }
}

void ProcessWithFilter(const Mat& in, const vector<Point>& points, Mat& out,
                       vector<size_t>& convex_hull) {
  const int kLowerThreshold = 150;
  const int kUpperThreshold = 230;
  Mat gray, blur, equalize, edges, edges_color;
  Mat mask(in.size(), CV_8U, Scalar::all(0));

  vector<Point> poly(convex_hull.size());
  for (size_t i = 0; i < convex_hull.size(); ++i) {
    poly[i] = points[convex_hull[i]];
  }

  out = in.clone();

  cvtColor(in, gray, COLOR_BGR2GRAY);
  medianBlur(gray, blur, 3);
  equalizeHist(blur, equalize);
  Canny(equalize, edges, kLowerThreshold, kUpperThreshold, 3, false);
  cvtColor(edges, edges_color, COLOR_GRAY2BGR);
  fillConvexPoly(mask, poly.data(), convex_hull.size(), Scalar(1));

  edges_color.copyTo(out, mask);
}

bool UpdatePointPosition(Point& position, Point& direction,
                         const Size& frame_size) {
  bool has_collision = false;
  position.x += direction.x;
  position.y += direction.y;

  if (position.x < 0) {
    position.x = 0;
    direction.x = -direction.x;
    has_collision = true;
  }
  if (position.x >= frame_size.width) {
    position.x = frame_size.width - 1;
    direction.x = -direction.x;
    has_collision = true;
  }

  if (position.y < 0) {
    position.y = 0;
    direction.y = -direction.y;
    has_collision = true;
  }
  if (position.y >= frame_size.height) {
    position.y = frame_size.height - 1;
    direction.y = -direction.y;
    has_collision = true;
  }

  return has_collision;
}

void OnMouse(int event, int x, int y, int, void* data) {
  auto event_data = static_cast<EventData*>(data);

  if (event == EVENT_LBUTTONDOWN) {
    event_data->point = Point(x, y);
  } else if (event == EVENT_LBUTTONUP) {
    int x_speed = (x - event_data->point.x) / 4;
    if (x_speed > kSpeedLimit) {
      x_speed = kSpeedLimit;
    } else if (x_speed < -kSpeedLimit) {
      x_speed = -kSpeedLimit;
    }

    int y_speed = (y - event_data->point.y) / 4;
    if (y_speed > kSpeedLimit) {
      y_speed = kSpeedLimit;
    } else if (y_speed < -kSpeedLimit) {
      y_speed = -kSpeedLimit;
    }

    event_data->points_position.emplace_back(x, y);
    event_data->points_direction.emplace_back(x_speed, y_speed);
  }
}

int main(int argc, const char** argv) {
  CommandLineParser parser(argc, argv, kOptions);
  parser.about(kAbout);

  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  int delay = parser.get<int>("delay");
  namedWindow(kWindowName, WINDOW_AUTOSIZE);
  moveWindow(kWindowName, 0, 0);

  EventData data;
  setMouseCallback(kWindowName, OnMouse, &data);

  Mat frame = imread(parser.get<String>(0));
  if (frame.empty()) {
    cout << "Failed to open image file '" + parser.get<String>(0) + "'."
         << endl;
    return 0;
  }

  int mode = 0;
  vector<size_t> convex_hull;

  while (true) {
    Mat out_image;
    for (size_t i = 0; i < data.points_position.size(); ++i) {
      UpdatePointPosition(data.points_position[i], data.points_direction[i],
                          frame.size());
    }
    ConvexHull(data.points_position, convex_hull);
    switch (mode) {
      case 0:
        Process(frame, out_image, data.points_position, convex_hull);
        break;
      case 1:
        ProcessWithFilter(frame, data.points_position, out_image, convex_hull);
        break;
    };

    imshow(kWindowName, out_image);

    int key = waitKey(delay) & 0x00FF;
    if (key == kEscapeKey) {
      break;
    }
    if (key == kSpacebarKey) {
      mode = (mode + 1) % 2;
    }
  }

  return 0;
}
