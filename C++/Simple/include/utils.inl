/*
 * Copyright (C) 2016 Cristian Peñaranda Cebrian
 */

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>

inline bool Utils::GetFile(const int &argc, char const *argv[],
                           unsigned char **data, size_t *data_size, int *rows,
                           int *columns) {
  bool result = argc == 3;
  if (result) {
    output_file_ = argv[2];
    result = (output_file_.compare(output_file_.length() - 4, 4, ".png") == 0);
    if (result) {
      std::string image_path = cv::samples::findFile(argv[1]);
      cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
      if (!img.empty()) {
        *rows = rows_ = img.rows;
        *columns = columns_ = img.cols;
        *data = new unsigned char[(img.rows + 2) * (img.cols + 2)];
        memset(*data, 0, img.cols + 2);
        *data_size = img.cols + 2;
        for (int i = 0; i < img.rows; ++i) {
          *(*data + *data_size) = 0;
          ++(*data_size);
          memcpy(*data + *data_size, img.data + (i * img.cols),
                 img.cols * sizeof(unsigned char));
          *data_size += img.cols * sizeof(unsigned char);
          *(*data + *data_size) = 0;
          ++(*data_size);
        }
        memset(*data + *data_size, 0, img.cols + 2);
        *data_size += img.cols + 2;
      } else {
        std::cout << "Could not read the image: " << image_path << std::endl;
        result = false;
      }
    }
  }

  if (!result) {
    std::cout
        << "To run the executable, you must indicate the image and the output:"
        << std::endl;
    std::cout << "  " << argv[0] << " <file> <output>.png" << std::endl;
  }
  return result;
}

inline void Utils::SaveFile(void *G_x, void *G_y, void *total) {
  cv::Mat image_to_save_G_x(rows_, columns_, CV_32F, G_x);
  cv::Mat image_to_save_G_y(rows_, columns_, CV_32F, G_y);
  cv::Mat image_to_save_total(rows_, columns_, CV_32F, total);
  if (cv::imwrite(output_file_, image_to_save_total)) {
    output_file_.erase(output_file_.length() - 4);
    std::string gx_output_file = output_file_ + "_gx.png";
    std::string gy_output_file = output_file_ + "_gy.png";
    if (!cv::imwrite(gx_output_file, image_to_save_G_x) ||
        !cv::imwrite(gy_output_file, image_to_save_G_y)) {
      std::cout << "Error saving G_x and G_y images" << std::endl;
    }
  } else {
    std::cout << "Error saving images" << std::endl;
  }
}