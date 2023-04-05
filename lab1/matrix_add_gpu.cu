// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.
// Update 2022: Changed to cudaDeviceSynchronize.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

const int N = 32; 
const int blocksize = 32; 

__global__ 
void add_matrix(float *a_d, float *b_d, float *c_d) 
{
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	c_d[Idx] = a_d[Idx] + b_d[Idx];
	
}

int main()
{	
	float *a_h, *b_h, *c_h; // pointer to the host arrays
	float *a_d, *b_d, *c_d; // pointer to the device arrays  

	const int size = N*N*sizeof(float);
	
	//allocate arrays on host
	a_h = (float *)malloc(size); 
	b_h = (float *)malloc(size); 
	c_h = (float *)malloc(size); 
	
	cudaMalloc((void **) &a_d, size); //allocate arrays on device
	cudaMalloc((void **) &b_d, size); //allocate arrays on device
	cudaMalloc((void **) &c_d, size); //allocate arrays on device
	
	// initializing the array
	for (int ii = 0; ii < N; ii++) {
		for (int jj = 0; jj < N; jj++) {
			a_h[ii+jj*N] = 10 + (float)ii;
			b_h[ii+jj*N] = (float)jj/N;
//			c_h[ii+jj*N] = 0; // initializing c_h array 
		}
	}
	
	// copy the array information from host to device
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice); 
// 	cudaMemcpy(c_d, c_h, size, cudaMemcpyHostToDevice); 
	
	dim3 dimBlock( blocksize, blocksize );
	dim3 dimGrid( N, 1 );
	
	add_matrix<<<dimGrid, dimBlock>>>(a_d, b_d, c_d);
	cudaDeviceSynchronize();
	
	// copy the array information from device to host
	cudaMemcpy( c_h, c_d, size, cudaMemcpyDeviceToHost ); 
	
	// free the mallocs
	cudaFree( a_d);
	cudaFree( b_d);
	cudaFree( c_d);
	
	//Data visulaization
	for (int ii = 0; ii < N; ii++) {
		for (int jj = 0; jj < N; jj++) {
			printf("%0.2f ", c_h[ii + jj*N]);
		}
	}
	printf("\n");

	printf("done\n");
	return EXIT_SUCCESS;
}
