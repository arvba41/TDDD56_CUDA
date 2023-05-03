#include <stdio.h> 

int main() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Shared Memory per Block: %zu \n",
            prop.sharedMemPerBlock);
        printf("  Total number of SMs: %d \n",
            prop.multiProcessorCount);
        printf("  Maximum number of threads per block: %d \n\n",
            prop.multiProcessorCount);
        printf("  Maximum Grid Size (x,y,z): %d, %d, %d\n",
            prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Maximum Threads diention (x,y,z): %d, %d, %d\n\n",
            prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    }
}