/*
 * Copyright (C) 2024 Cristian Peñaranda Cebrian
 */

#pragma once

#include <cuda_runtime.h>

#include <opencv2/imgcodecs.hpp>
#include <string>

class Utils {
 private:
  int rows_;
  int columns_;
  std::string output_file_;

  int *device_sobel_x_operator_;
  int *device_sobel_y_operator_;
  unsigned char *device_image_;
  size_t image_size_;
  float *host_G_x_, *device_G_x_, *host_G_y_, *device_G_y_;
  float *host_total_, *device_total_;

  bool SavingImageToDeviceMemory(const cv::Mat &host_image);

  bool LocatingMemory();

  bool RemoveMemories();

 public:
  bool GetFile(const int &argc, char const *argv[]);

  bool SobelFilter();

  bool SaveFile();

  Utils();

  ~Utils();
};

#include <utils.inl>
