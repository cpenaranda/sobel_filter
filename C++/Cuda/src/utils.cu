/*
 * Copyright (C) 2024 Cristian Peñaranda Cebrian
 */

#include <iostream>
#include <string>
#include <utils.cuh>

__global__ void SolbeFilter(int *x, int *y, unsigned char *image, float *G_x,
                            float *G_y, float *total, size_t size,
                            int columns) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int done;
  while (index < size) {
    done = index + ((index / columns) * 2);
    G_x[index] = 0;
    G_y[index] = 0;
    for (int i = 0; i < 3; ++i) {
      G_x[index] += x[(i * 3) + 0] * image[done] +
                    x[(i * 3) + 1] * image[done + 1] +
                    x[(i * 3) + 2] * image[done + 2];
      G_y[index] += y[(i * 3) + 0] * image[done] +
                    y[(i * 3) + 1] * image[done + 1] +
                    y[(i * 3) + 2] * image[done + 2];
      done += columns + 2;
    }
    total[index] = sqrt((G_x[index] * G_x[index]) + (G_y[index] * G_y[index]));
    index += (gridDim.x * blockDim.x);
  }
}

bool Utils::SobelFilter() {
  SolbeFilter<<<2048, 64, 0, nullptr>>>(
      device_sobel_x_operator_, device_sobel_y_operator_, device_image_,
      device_G_x_, device_G_y_, device_total_, rows_ * columns_, columns_);
  cudaError_t error =
      cudaMemcpy(host_G_x_, device_G_x_, sizeof(float) * rows_ * columns_,
                 cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return false;
  error = cudaMemcpy(host_G_y_, device_G_y_, sizeof(float) * rows_ * columns_,
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return false;
  error = cudaMemcpy(host_total_, device_total_, sizeof(float) * rows_ * columns_,
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return false;
  return true;
}

Utils::Utils() {}

Utils::~Utils() {}