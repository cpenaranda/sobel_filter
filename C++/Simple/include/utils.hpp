/*
 * Copyright (C) 2016 Cristian Peñaranda Cebrian
 */

#pragma once

#include <array>
#include <string>

class Utils {
 private:
  static int rows_;
  static int columns_;
  static std::string output_file_;

 public:
  static constexpr std::array<std::array<float, 3>, 3> sobel_x_operator_ = {
      {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}}};
  static constexpr std::array<std::array<float, 3>, 3> sobel_y_operator_ = {
      {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}}};

  static bool GetFile(const int &argc, char const *argv[], unsigned char **data,
                      size_t *data_size, int *rows, int *columns);

  static void SaveFile(void *G_x, void *G_y, void *total);
};

#include <utils.inl>
