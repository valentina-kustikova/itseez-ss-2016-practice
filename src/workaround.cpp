#include "workaround.hpp"

void MatrixProcessor::Threshold(uchar* const data, const int width,
    const int height, const int threshold) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      data[row * width + col] = (data[row * width + col] < threshold) ? 0 :
        data[row * width + col];
    }
  }
}
