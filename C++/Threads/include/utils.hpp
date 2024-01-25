/*
 * Copyright (C) 2024 Cristian Peñaranda Cebrian
 */

#pragma once

#include <array>
#include <condition_variable>  // NOLINT
#include <mutex>               // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>

class Utils {
 private:
  static int rows_;
  static int columns_;
  static std::string output_file_;

  static size_t size_;
  static std::mutex mutex_;
  static std::condition_variable condition_;
  static std::condition_variable condition_parent_;
  static float *G_x_;
  static float *G_y_;
  static float *total_;
  static unsigned char *image_;
  static bool threads_alive_;
  static std::vector<std::thread> threads_;

  static constexpr std::array<std::array<float, 3>, 3> sobel_x_operator_ = {
      {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}}};
  static constexpr std::array<std::array<float, 3>, 3> sobel_y_operator_ = {
      {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}}};

  static void ThreadBehavior();
  static void CreateThreads(const int &number_of_threads);
  static void RemoveThreads();
  static void WaitThreads();

 public:
  static bool GetFile(const int &argc, char const *argv[], unsigned char **data,
                      size_t *data_size, int *rows, int *columns);

  static void Initialize(unsigned char *const image, float *const G_x,
                         float *const G_y, float *const total);

  static void SaveFile(void *G_x, void *G_y, void *total);
};

#include <utils.inl>
