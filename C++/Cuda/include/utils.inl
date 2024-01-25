/*
 * Copyright (C) 2024 Cristian Peñaranda Cebrian
 */

inline bool Utils::SavingImageToDeviceMemory(const cv::Mat &host_image) {
  rows_ = host_image.rows;
  columns_ = host_image.cols;
  cudaError_t error =
      cudaMalloc(&device_image_, sizeof(unsigned char) * (host_image.rows + 2) *
                                     (host_image.cols + 2));
  if (error != cudaSuccess) return false;
  error = cudaMemset(device_image_, 0, host_image.cols + 2);
  if (error != cudaSuccess) return false;
  image_size_ = host_image.cols + 2;
  for (int i = 0; i < host_image.rows; ++i) {
    error = cudaMemset(device_image_ + image_size_, 0, 1);
    if (error != cudaSuccess) return false;
    ++image_size_;
    error = cudaMemcpy(
        device_image_ + image_size_, host_image.data + (i * host_image.cols),
        host_image.cols * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return false;
    image_size_ += host_image.cols * sizeof(unsigned char);
    error = cudaMemset(device_image_ + image_size_, 0, 1);
    if (error != cudaSuccess) return false;
    ++image_size_;
  }
  error = cudaMemset(device_image_ + image_size_, 0, host_image.cols + 2);
  if (error != cudaSuccess) return false;
  image_size_ += host_image.cols + 2;
  return true;
}

inline bool Utils::LocatingMemory() {
  cudaError_t error = cudaHostAlloc(
      &host_G_x_, sizeof(float) * rows_ * columns_, cudaHostAllocDefault);
  if (error != cudaSuccess) return false;
  error = cudaHostAlloc(&host_G_y_, sizeof(float) * rows_ * columns_,
                        cudaHostAllocDefault);
  if (error != cudaSuccess) return false;
  error = cudaHostAlloc(&host_total_, sizeof(float) * rows_ * columns_,
                        cudaHostAllocDefault);
  if (error != cudaSuccess) return false;
  error = cudaMalloc(&device_G_x_, sizeof(float) * rows_ * columns_);
  if (error != cudaSuccess) return false;
  error = cudaMalloc(&device_G_y_, sizeof(float) * rows_ * columns_);
  if (error != cudaSuccess) return false;
  error = cudaMalloc(&device_total_, sizeof(float) * rows_ * columns_);
  if (error != cudaSuccess) return false;

  int host_sobel_x_operator[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
  int host_sobel_y_operator[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  error = cudaMalloc(&device_sobel_x_operator_, sizeof(int) * 9);
  if (error != cudaSuccess) return false;
  error = cudaMemcpy(device_sobel_x_operator_, host_sobel_x_operator,
                     sizeof(int) * 9, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return false;
  error = cudaMalloc(&device_sobel_y_operator_, sizeof(int) * 9);
  if (error != cudaSuccess) return false;
  error = cudaMemcpy(device_sobel_y_operator_, host_sobel_y_operator,
                     sizeof(int) * 9, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) return false;
  return true;
}

inline bool Utils::RemoveMemories() {
  cudaError_t error = cudaFree(device_image_);
  if (error != cudaSuccess) return false;
  error = cudaFree(device_sobel_x_operator_);
  if (error != cudaSuccess) return false;
  error = cudaFree(device_sobel_y_operator_);
  if (error != cudaSuccess) return false;
  error = cudaFree(device_G_x_);
  if (error != cudaSuccess) return false;
  error = cudaFree(device_G_y_);
  if (error != cudaSuccess) return false;
  error = cudaFree(device_total_);
  if (error != cudaSuccess) return false;
  error = cudaFreeHost(host_G_x_);
  if (error != cudaSuccess) return false;
  error = cudaFreeHost(host_G_y_);
  if (error != cudaSuccess) return false;
  error = cudaFreeHost(host_total_);
  if (error != cudaSuccess) return false;
  return true;
}

inline bool Utils::GetFile(const int &argc, char const *argv[]) {
  bool result = argc == 3;
  if (result) {
    output_file_ = argv[2];
    result = (output_file_.compare(output_file_.length() - 4, 4, ".png") == 0);
    if (result) {
      std::string image_path = cv::samples::findFile(argv[1]);
      cv::Mat host_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
      if (!host_image.empty() && SavingImageToDeviceMemory(host_image)) {
        if (!LocatingMemory()) {
          std::cout << "Error locating gpu memory" << std::endl;
          result = false;
        }
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

inline bool Utils::SaveFile() {
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return false;
  cv::Mat image_to_save_G_x(rows_, columns_, CV_32F, host_G_x_);
  cv::Mat image_to_save_G_y(rows_, columns_, CV_32F, host_G_y_);
  cv::Mat image_to_save_total(rows_, columns_, CV_32F, host_total_);
  if (cv::imwrite(output_file_, image_to_save_total)) {
    output_file_.erase(output_file_.length() - 4);
    std::string gx_output_file = output_file_ + "_gx.png";
    std::string gy_output_file = output_file_ + "_gy.png";
    if (!cv::imwrite(gx_output_file, image_to_save_G_x) ||
        !cv::imwrite(gy_output_file, image_to_save_G_y)) {
      std::cout << "Error saving G_x and G_y images" << std::endl;
      return false;
    }
  } else {
    std::cout << "Error saving images" << std::endl;
    return false;
  }
  if (!RemoveMemories()) {
    std::cout << "Error removing memories" << std::endl;
    return false;
  }
  return true;
}