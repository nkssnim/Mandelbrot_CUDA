# Works with 3.14 and tested through 3.29
cmake_minimum_required(VERSION 3.14...3.29)

# Project name and a few useful settings. Other commands can pick up the results
project(
  Mandelbrot_CUDA
  VERSION 0.1
  DESCRIPTION "Mandelbrot set generation accelerated with CUDA"
  LANGUAGES CXX)

# The compiled library code is here
add_subdirectory(src)