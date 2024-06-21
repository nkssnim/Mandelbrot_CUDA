#include <cuComplex.h>
#include <cuda/std/chrono>
#include "window.h"
#include "save_image.h"
#include "utils.h"
#include <vector>
#include <iostream>
#include <stdio.h>

#define Y_MIN 0
#define Y_MAX 1200
#define X_MIN 0
#define X_MAX 1200
#define ITER_MAX 500
#define FRACT_Y_MIN -1.7
#define FRACT_Y_MAX 1.7
#define FRACT_X_MIN -2.2
#define FRACT_X_MAX 1.2
#define TRIP_FRACT_Y_MIN -1.5
#define TRIP_FRACT_Y_MAX 1.5
#define TRIP_FRACT_X_MIN -1.5
#define TRIP_FRACT_X_MAX 1.5

// Use an alias to simplify the use of complex type
using Complex = cuDoubleComplex;

// Convert a pixel coordinate to the complex domain
__device__
Complex scale(Complex c, bool trip) {
	Complex aux{};
	if (trip){
		aux = make_cuDoubleComplex(cuCreal(c) / (double)(X_MAX-X_MIN) * ((TRIP_FRACT_X_MAX)-(TRIP_FRACT_X_MIN)) + TRIP_FRACT_X_MIN,
			cuCimag(c) / (double)(Y_MAX-Y_MIN) * ((TRIP_FRACT_Y_MAX)-(TRIP_FRACT_Y_MIN)) + TRIP_FRACT_Y_MIN);
	}
	else{
		aux = make_cuDoubleComplex(cuCreal(c) / (double)(X_MAX-X_MIN) * ((FRACT_X_MAX)-(FRACT_X_MIN)) + FRACT_X_MIN,
			cuCimag(c) / (double)(Y_MAX-Y_MIN) * ((FRACT_Y_MAX)-(FRACT_Y_MIN)) + FRACT_Y_MIN);
	}
	return aux;
}

__device__
Complex func(Complex z, Complex c, bool trip){
	if (trip){
		return cuCadd(cuCmul(z, cuCmul(z,z)), c);
	}
	return cuCadd(cuCmul(z,z), c);
}

// Check if a point is in the set or escapes to infinity, return the number if iterations
__device__
int escape(Complex c, bool trip) {
	Complex z = make_cuDoubleComplex(0.0, 0.0);
	int iter = 0;

	while (cuCabs(z) < 2.0 && iter < ITER_MAX) {
		z = func(z, c, trip);
		iter++;
	}

	return iter;
}

// Loop over each pixel from our image and check if the points associated with this pixel escape to infinity
__global__
void get_number_iterations(int* colors, bool trip) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < (X_MAX - X_MIN) && col < (Y_MAX - Y_MIN)){
		Complex c = make_cuDoubleComplex((double)row, (double)col);
		c = scale(c, trip);
		colors[col * (X_MAX - X_MIN) + row] = escape(c, trip);
	}
	// for(int i = scr.y_min(); i < scr.y_max(); ++i) {
	// 	for(int j = scr.x_min(); j < scr.x_max(); ++j) {
	// 		Complex c((double)j, (double)i);
	// 		c = scale(scr, fract, c);
	// 		colors[k] = escape(c, iter_max, func);
	// 		k++;
	// 	}
	// 	if(progress < (int)(i*100.0/scr.y_max())){
	// 		progress = (int)(i*100.0/scr.y_max());
	// 		std::cout << progress << "%\n";
	// 	}
	// }
    
}

// void fractal(window<int> &scr, window<double> &fract, int iter_max, std::vector<int> &colors,
// 	const std::function<Complex( Complex, Complex)> &func, const char *fname, bool smooth_color) {
// 	auto start = std::chrono::high_resolution_clock::now();
// 	get_number_iterations(scr, fract, iter_max, colors, func);
// 	auto end = std::chrono::high_resolution_clock::now();
// 	std::cout << "Time to generate " << fname << " = " << std::chrono::duration <double, std::milli> (end - start).count() << " [ms]" << std::endl;

// 	// Save (show) the result as an image
// 	plot(scr, colors, iter_max, fname, smooth_color);
// }

void fractal(window<int> &scr, window<double> &fract, int* colors, const char *fname, bool smooth_color, bool trip){
	//auto start = std::chrono::high_resolution_clock::now();
	dim3 threads_per_block (16, 16, 1);
	dim3 number_of_blocks (((Y_MAX - Y_MIN) / threads_per_block.x), ((X_MAX - X_MIN) / threads_per_block.y), 1);
	get_number_iterations<<<number_of_blocks, threads_per_block>>>(colors, trip);
	cudaDeviceSynchronize();
	std::vector<int> colorVec(scr.size());
	cudaMemcpy(&colorVec[0], colors, scr.size() * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	//auto end = std::chrono::high_resolution_clock::now();
	plot(scr, colorVec, ITER_MAX, fname, smooth_color);
}

void mandelbrot() {
	// Define the size of the image
	window<int> scr(X_MIN, X_MAX, Y_MIN, Y_MAX);
	// The domain in which we test for points
	window<double> fract(FRACT_X_MIN, FRACT_X_MAX, FRACT_Y_MIN, FRACT_Y_MAX);

	int* colors;
	cudaMallocManaged(&colors, (Y_MAX - Y_MIN) * (X_MAX - X_MIN) * sizeof(int));
	cudaDeviceSynchronize();	


	// The function used to calculate the fractal
	//auto func = [] (Complex z, Complex c) -> Complex {return z * z + c; };

	const char *fname = "mandelbrot_cuda.png";
	bool smooth_color = true;
	//std::vector<int> colors(scr.size());

	// Experimental zoom (bugs ?). This will modify the fract window (the domain in which we calculate the fractal function) 
	//zoom(1.0, -1.225, -1.22, 0.15, 0.16, fract); //Z2
	
	fractal(scr, fract, colors, fname, smooth_color, false);
	cudaDeviceSynchronize();
	cudaFree(colors);
}

void triple_mandelbrot() {
	// Define the size of the image
	window<int> scr(X_MIN, X_MAX, Y_MIN, Y_MAX);
	// The domain in which we test for points
	window<double> fract(TRIP_FRACT_X_MIN, TRIP_FRACT_X_MAX, TRIP_FRACT_Y_MIN, TRIP_FRACT_Y_MAX);

	int* colors;
	cudaMallocManaged(&colors, (Y_MAX - Y_MIN) * (X_MAX - X_MIN) * sizeof(int));
	cudaDeviceSynchronize();

	const char *fname = "triple_mandelbrot_cuda.png";
	bool smooth_color = true;

	fractal(scr, fract, colors, fname, smooth_color, true);
	cudaDeviceSynchronize();
	cudaFree(colors);
}

int main() {

	mandelbrot();
	triple_mandelbrot();

	return 0;
}
