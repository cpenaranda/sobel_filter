/*
 * Copyright (C) 2024 Cristian Peñaranda Cebrian
 */

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>

inline void Utils::ThreadBehavior() {
  size_t index = 0, done = 0;
  while (threads_alive_) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(
          lock, [] { return !threads_alive_ || size_ < rows_ * columns_; });
      if (!threads_alive_)
        break;
      else {
        index = size_;
        done = index + ((size_ / columns_) * 2);
        ++size_;
      }
    }
    G_x_[index] = 0;
    G_y_[index] = 0;
    for (int i = 0; i < 3; ++i) {
      G_x_[index] += sobel_x_operator_[i][0] * image_[done] +
                     sobel_x_operator_[i][1] * image_[done + 1] +
                     sobel_x_operator_[i][2] * image_[done + 2];
      G_y_[index] += sobel_y_operator_[i][0] * image_[done] +
                     sobel_y_operator_[i][1] * image_[done + 1] +
                     sobel_y_operator_[i][2] * image_[done + 2];
      done += columns_ + 2;
    }
    total_[index] =
        std::sqrt(std::pow(G_x_[index], 2) + std::pow(G_y_[index], 2));
    if (size_ == rows_ * columns_) condition_parent_.notify_one();
  }
}

inline void Utils::CreateThreads(const int &number_of_threads) {
  size_ = rows_ * columns_;
  threads_.clear();
  threads_alive_ = true;
  for (int thread = 0; thread < number_of_threads; ++thread) {
    threads_.push_back(std::thread(&ThreadBehavior));
  }
}

inline void Utils::RemoveThreads() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    threads_alive_ = false;
    condition_.notify_all();
  }
  for (auto &thread : threads_) {
    if (thread.joinable()) thread.join();
  }
  threads_.clear();
}
inline void Utils::WaitThreads() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_parent_.wait(
        lock, [] { return !threads_alive_ || size_ == rows_ * columns_; });
  }
  RemoveThreads();
}

inline bool Utils::GetFile(const int &argc, char const *argv[],
                           unsigned char **data, size_t *data_size, int *rows,
                           int *columns) {
  bool result = argc == 4;
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
        CreateThreads(atoi(argv[3]));
      } else {
        std::cout << "Could not read the image: " << image_path << std::endl;
        result = false;
      }
    }
  }

  if (!result) {
    std::cout << "To run the executable, you must indicate the image, the "
                 "output and the number of threads:"
              << std::endl;
    std::cout << "  " << argv[0] << " <file> <output>.png <number_of_threads>"
              << std::endl;
  }
  return result;
}

inline void Utils::Initialize(unsigned char *const image, float *const G_x,
                              float *const G_y, float *const total) {
  image_ = image;
  G_x_ = G_x;
  G_y_ = G_y;
  total_ = total;
  std::lock_guard<std::mutex> lock(mutex_);
  size_ = 0;
  condition_.notify_all();
}

inline void Utils::SaveFile(void *G_x, void *G_y, void *total) {
  WaitThreads();
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
