/*
 * Copyright (C) 2024 Cristian Peñaranda Cebrian
 */

#include <iostream>
#include <utils.cuh>

int main(int argc, char const *argv[]) {
  int error = EXIT_SUCCESS;
  Utils *utility = new Utils();
  if (utility->GetFile(argc, argv)) {
    utility->SobelFilter();
    utility->SaveFile();
  } else {
    error = EXIT_FAILURE;
  }
  delete utility;
  return error;
}
