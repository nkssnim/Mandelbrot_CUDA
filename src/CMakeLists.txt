# Notice name prefix of this variable, set by CMake according
# to value given with "project()" in the root CMakeLists.txt.

add_executable(mandelbrot save_image.cpp utils.cpp mandel.cu)
add_executable(mandelbrot_cpu save_image.cpp utils.cpp mandel.cpp)
target_include_directories(mandelbrot_cpu PRIVATE ${Mandelbrot_CUDA_SOURCE_DIR}/include)
target_include_directories(mandelbrot_cpu PRIVATE ${Mandelbrot_CUDA_SOURCE_DIR}/include/FreeImage)
target_link_libraries(mandelbrot_cpu ${Mandelbrot_CUDA_SOURCE_DIR}/include/FreeImage/FreeImage.lib)
target_include_directories(mandelbrot PRIVATE ${Mandelbrot_CUDA_SOURCE_DIR}/include)
target_include_directories(mandelbrot PRIVATE ${Mandelbrot_CUDA_SOURCE_DIR}/include/FreeImage)
target_link_libraries(mandelbrot ${Mandelbrot_CUDA_SOURCE_DIR}/include/FreeImage/FreeImage.lib)
