/*
 * Copyright (C) 2024 Cristian Peñaranda Cebrian
 */

#include <cmath>
#include <iostream>
#include <utils.hpp>

void SobelFilter(const unsigned char *const image, const size_t &data_size,
                 const int &rows, const int &columns, float **G_x, float **G_y,
                 float **total) {
  *G_x = new float[rows * columns];
  *G_y = new float[rows * columns];
  *total = new float[rows * columns];
  size_t done = 0;
  size_t index = 0;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < columns; ++c) {
      done = r * (columns + 2) + c;
      index = r * columns + c;
      (*G_x)[index] = 0;
      (*G_y)[index] = 0;
      for (int i = 0; i < 3; ++i) {
        (*G_x)[index] += Utils::sobel_x_operator_[i][0] * image[done] +
                         Utils::sobel_x_operator_[i][1] * image[done + 1] +
                         Utils::sobel_x_operator_[i][2] * image[done + 2];
        (*G_y)[index] += Utils::sobel_y_operator_[i][0] * image[done] +
                         Utils::sobel_y_operator_[i][1] * image[done + 1] +
                         Utils::sobel_y_operator_[i][2] * image[done + 2];
        done += columns + 2;
      }
      (*total)[index] =
          std::sqrt(std::pow((*G_x)[index], 2) + std::pow((*G_y)[index], 2));
    }
  }
}

int main(int argc, char const *argv[]) {
  unsigned char *image;
  size_t data_size = 0;
  int rows = 0, columns = 0;
  int error = EXIT_SUCCESS;
  if (Utils::GetFile(argc, argv, &image, &data_size, &rows, &columns)) {
    float *G_x, *G_y, *total;
    SobelFilter(image, data_size, rows, columns, &G_x, &G_y, &total);
    Utils::SaveFile(G_x, G_y, total);
    delete[] G_x;
    delete[] G_y;
    delete[] total;
    delete[] image;
  } else {
    error = EXIT_FAILURE;
  }
  return error;
}
