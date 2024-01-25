/*
 * Copyright (C) 2024 Cristian Peñaranda Cebrian
 */

#include <cmath>
#include <iostream>
#include <utils.hpp>

void SobelFilter(unsigned char *const image, const size_t &data_size,
                 const int &rows, const int &columns, float **G_x, float **G_y,
                 float **total) {
  *G_x = new float[rows * columns];
  *G_y = new float[rows * columns];
  *total = new float[rows * columns];
  Utils::Initialize(image, *G_x, *G_y, *total);
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
