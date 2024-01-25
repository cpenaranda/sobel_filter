/*
 * Copyright (C) 2024 Cristian Peñaranda Cebrian
 */

#include <utils.hpp>

int Utils::rows_ = 0;
int Utils::columns_ = 0;
std::string Utils::output_file_;  // NOLINT

size_t Utils::size_ = 0;
std::mutex Utils::mutex_;
std::condition_variable Utils::condition_;
std::condition_variable Utils::condition_parent_;
float *Utils::G_x_;
float *Utils::G_y_;
float *Utils::total_;
unsigned char *Utils::image_;
bool Utils::threads_alive_ = true;
std::vector<std::thread> Utils::threads_;
