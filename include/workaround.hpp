#pragma once

#include <memory>
#include <string>

#include "opencv2/core/core.hpp"

class MatrixProcessor {
 public:
   virtual void Threshold(uchar* const data, const int width, 
     const int height, const int threshold) = 0;
};

class MatrixProcessorImpl : public MatrixProcessor {
public:
  virtual void Threshold(uchar* const data, const int width,
    const int height, const int threshold);
};
