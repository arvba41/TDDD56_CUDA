// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.
// Update 2022: Changed to cudaDeviceSynchronize.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <math.h>

const int N = 256; 
const int blocksize = 256; 

__global__ 
void square_array(float *a_d) 
{
	a_d[threadIdx.x] = sqrt(a_d[threadIdx.x]);
}

int main()
{	
	float *a_h, *a_d; // pointer to the host and device arrays 

	const int size = N*sizeof(float);
	
	a_h = (float *)malloc(size); //allocate arrays on host
	cudaMalloc((void **) &a_d, size); //allocate arrays on device
	
	// initializing the array
	for (int ii = 0; ii < N; ii++) {
		a_h[ii] = (float)ii; 
	}
	
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice); // copy the array information from host to device
	
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	square_array<<<dimGrid, dimBlock>>>(a_d);
	cudaDeviceSynchronize();
	cudaMemcpy( a_h, a_d, size, cudaMemcpyDeviceToHost ); 
	cudaFree( a_d );
	
	for (int ii = 0; ii < N; ii++)
		printf("%f ", a_h[ii]);
	printf("\n");
//	delete[] a_h;
	printf("done\n");
	return EXIT_SUCCESS;
}
