# Works with 3.14 and tested through 3.29
cmake_minimum_required(VERSION 3.14...3.29)

# Project name and a few useful settings. Other commands can pick up the results
project(
  Mandelbrot_CUDA
  VERSION 0.1
  DESCRIPTION "Mandelbrot set generation accelerated with CUDA"
  LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# The compiled library code is here
add_subdirectory(src)