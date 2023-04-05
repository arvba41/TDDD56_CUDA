// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.
// Update 2022: Changed to cudaDeviceSynchronize.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include "milli.c"

const int N = 1024; 
const int blocksize = 1024; 

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

	float thetime; // declating the pointer to store time result
	
	const int size = N*N*sizeof(float);
	
	cudaEvent_t myEvent1; //CUDA event 1
	cudaEvent_t myEvent2; //CUDA event 2
	
	cudaEventCreate(&myEvent1); // CUDA event initialization
	cudaEventCreate(&myEvent2); // CUDA event initialization
		
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
		}
	}
	
	// copy the array information from host to device
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice); 
	
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( N, 1 );
	
	cudaEventRecord(myEvent1, 0); // inserting event into the cuda stream
	ResetMilli(); // inserting cpu timer (reset)
	
	add_matrix<<<dimGrid, dimBlock>>>(a_d, b_d, c_d);
	
	cudaEventRecord(myEvent2, 0); // inserting event into the cuda stream
	int thetime_cpu = GetMicroseconds(); // get cpu time
	
	cudaDeviceSynchronize();
	cudaEventSynchronize(myEvent2);
	
	cudaEventElapsedTime(&thetime, myEvent1, myEvent2);
	
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


	printf("the time taken for the event is %f us GPU event timer. \n", thetime*1000);

	
	printf("the time taken for the event is %d us CPU timer. \n", thetime_cpu);

	printf("done\n");
	return EXIT_SUCCESS;
}
