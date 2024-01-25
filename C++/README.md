# Sobel Filter - C++ Version

Different solutions using C++.

# Simple
It applies the Sobel filter using one thread.

# Threads
A multi-thread solution where the user can indicate the number of threads to apply the Sobel filter.

# Cuda
A simple solution where the GPU is used to apply the Sobel filter.

# How to compile

```
mkdir build; cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=<opencv_build> -DCUDA_TOOLKIT_ROOT_DIR=<cuda_dir> ..
cmake --build . --config Release --target all
```

# How to run

```
bin/sobel_filter_simple <file> <output>.png
bin/sobel_filter_threads <file> <output>.png <number_of_threads>
bin/sobel_filter_cuda <file> <output>.png
```