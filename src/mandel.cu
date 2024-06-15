#include <cuComplex.h>
#include <cuda/std/chrono>
#include "window.h"
#include "save_image.h"
#include "utils.h"
#include <vector>

#define Y_MIN 0
#define Y_MAX 1200
#define X_MIN 0
#define X_MAX 1200
#define ITER_MAX 500
#define FRACT_Y_MIN -1.7
#define FRACT_Y_MAX 1.7
#define FRACT_X_MIN -2.2
#define FRACT_X_MAX 1.2

// Use an alias to simplify the use of complex type
using Complex = cuDoubleComplex;

// Convert a pixel coordinate to the complex domain
__device__
Complex scale(Complex c) {
	Complex aux = make_cuDoubleComplex(cuCreal(c) / (double)(X_MAX-X_MIN) * ((FRACT_X_MAX)-(FRACT_X_MIN)) + FRACT_X_MIN,
		cuCimag(c) / (double)(Y_MAX-Y_MIN) * ((FRACT_Y_MAX)-(FRACT_Y_MIN)) + FRACT_Y_MIN);
	return aux;
}

__device__
Complex func(Complex z, Complex c){
	return cuCadd(cuCmul(z,z), c);
}

// Check if a point is in the set or escapes to infinity, return the number if iterations
__device__
int escape(Complex c) {
	Complex z = make_cuDoubleComplex(0.0, 0.0);
	int iter = 0;

	while (cuCabs(z) < 2.0 && iter < ITER_MAX) {
		z = func(z, c);
		iter++;
	}

	return iter;
}

// Loop over each pixel from our image and check if the points associated with this pixel escape to infinity
__global__
void get_number_iterations(int* colors) {
	int k = 0, progress = -1;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < (X_MAX - X_MIN) && col < (Y_MAX - Y_MIN)){
		for (int i = 0; i < (Y_MAX - Y_MIN); i++){
			Complex c = make_cuDoubleComplex((double)row, (double)col);
			c = scale(c);
			colors[k++] = escape(c);
		}
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

void fractal(window<int> &scr, window<double> &fract, int* colors, const char *fname, bool smooth_color){
	//auto start = std::chrono::high_resolution_clock::now();
	dim3 threads_per_block (16, 16, 1);
	dim3 number_of_blocks (((Y_MAX - Y_MIN) / threads_per_block.x) + 1, ((X_MAX - X_MIN) / threads_per_block.y) + 1, 1);
	get_number_iterations<<<number_of_blocks, threads_per_block>>>(colors);
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

	const char *fname = "mandelbrot.png";
	bool smooth_color = true;
	//std::vector<int> colors(scr.size());

	// Experimental zoom (bugs ?). This will modify the fract window (the domain in which we calculate the fractal function) 
	//zoom(1.0, -1.225, -1.22, 0.15, 0.16, fract); //Z2
	
	fractal(scr, fract, colors, fname, smooth_color);
	cudaDeviceSynchronize();
	cudaFree(colors);
}

// void triple_mandelbrot() {
// 	// Define the size of the image
// 	window<int> scr(0, 1200, 0, 1200);
// 	// The domain in which we test for points
// 	window<double> fract(-1.5, 1.5, -1.5, 1.5);

// 	// The function used to calculate the fractal
// 	auto func = [] (Complex z, Complex c) -> Complex {return z * z * z + c; };

// 	int iter_max = 500;
// 	const char *fname = "triple_mandelbrot.png";
// 	bool smooth_color = true;
// 	std::vector<int> colors(scr.size());

// 	fractal(scr, fract, iter_max, colors, func, fname, smooth_color);
// }

int main() {

	mandelbrot();
	//triple_mandelbrot();

	return 0;
}
