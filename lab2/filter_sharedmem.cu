// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -arch=sm_30 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
// or (multicore lab)
// nvcc filter.cu -c -arch=sm_20 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -L/usr/local/cuda/lib64 -lcudart -lglut -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may come
// but I call this version 1.0b2.
// 2017-12-04: Two fixes: Added command-lines (above), fixed a bug in computeImages
// that allocated too much memory. b3
// 2017-12-04: More fixes: Tightened up the kernel with edge clamping.
// Less code, nicer result (no borders). Cleaned up some messed up X and Y. b4
// 2022-12-07: A correction for a deprecated function.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"

// Use these for setting shared memory size.
#define maxKernelSizeX 10
#define maxKernelSizeY 10

const int blockSize = 32; // #threads per block

/* This is the original code -----------
__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{ 
  // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int dy, dx;
  	unsigned int sumx, sumy, sumz;

  	int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!
	
	if (x < imagesizex && y < imagesizey) // If inside image
	{
// Filter kernel (simple box filter)
	sumx=0;sumy=0;sumz=0;
	for(dy=-kernelsizey;dy<=kernelsizey;dy++)
		for(dx=-kernelsizex;dx<=kernelsizex;dx++)	
		{
			// Use max and min to avoid branching!
			int yy = min(max(y+dy, 0), imagesizey-1);
			int xx = min(max(x+dx, 0), imagesizex-1);
			
			sumx += image[((yy)*imagesizex+(xx))*3+0];
			sumy += image[((yy)*imagesizex+(xx))*3+1];
			sumz += image[((yy)*imagesizex+(xx))*3+2];
		}
	out[(y*imagesizex+x)*3+0] = sumx/divby;
	out[(y*imagesizex+x)*3+1] = sumy/divby;
	out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
} 
*/

// function for the filter with shared memory
__global__ void filter_sharedmem(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{  
    // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// normalization factor
	int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!

    // int dy, dx;
    unsigned int sumx, sumy, sumz;

	/* ---Shared memeoty part-- */
    // Max #Threads per block (3 chars per pixel)
    __shared__ unsigned char imgArray[blockSize*blockSize*3];
  
    imgArray[(threadIdx.y*blockDim.x+threadIdx.x)*3+0] = image[(y*imagesizex+x)*3+0];
    imgArray[(threadIdx.y*blockDim.x+threadIdx.x)*3+1] = image[(y*imagesizex+x)*3+1];
	imgArray[(threadIdx.y*blockDim.x+threadIdx.x)*3+2] = image[(y*imagesizex+x)*3+2];

    // synchronize between threads 
    __syncthreads();

	if (x < imagesizex && y < imagesizey) // If inside image
	{
        // Filter kernel (simple box filter)
        sumx=0;sumy=0;sumz=0;
        //Loop across image, from -filtersize to +filtersize 
        //Default is 2,2 ( so it is a 5x5 box filter.) -2   -   2 (-2,-1,0,1,2)
        for(int dy=-kernelsizey;dy<=kernelsizey;dy++) {  // y direction
            // both directions, total size is 5x5 = 25 (DivBy) variable
            for(int dx=-kernelsizex;dx<=kernelsizex;dx++) { // x direction
                
				// creating a new indexing variable becuase we not have a shared memory bank
				int index = (threadIdx.y*blockDim.x+threadIdx.x) + dy*blockDim.x + dx; 

				/* --- the seris of checks --- 
				1. Ensuring that the index is withing the block bounds 
				2. The shared memory for one element near and at the corners are ignored
				*/
                if(!((index < 0) || (index > (blockSize*blockSize-1)) || (dx > (blockDim.x-kernelsizex)) || (dy > (blockDim.y-kernelsizey)))) {
                    
                    sumx += imgArray[index*3+0];
                    sumy += imgArray[index*3+1];
                    sumz += imgArray[index*3+2];
                    
                } 
				else { // Outside of block row
                    int yy = min(max(y+dy, 0), imagesizey-1);
                    int xx = min(max(x+dx, 0), imagesizex-1);

                    sumx += image[((yy)*imagesizex+(xx))*3+0];
                    sumy += image[((yy)*imagesizex+(xx))*3+1];
                    sumz += image[((yy)*imagesizex+(xx))*3+2];
                }
            }
        }

        // printf("inverse is %f \n", invDivBy);
        // printf("Inverse x is %f", sumx*invDivBy);
        // printf(" x is %f", sumx/divby);

        out[(y*imagesizex+x)*3+0] = sumx/divby;
        out[(y*imagesizex+x)*3+1] = sumy/divby;
		out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}


// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);

	dim3 dimBlock(blockSize,blockSize);
	dim3 dimgrid(imagesizex/blockSize,imagesizey/blockSize);

	// creating cuda events for timing
	cudaEvent_t beforeEvent;
	cudaEvent_t afterEvent;
	float theTime;
	cudaEventCreate(&beforeEvent);
	cudaEventCreate(&afterEvent);
	cudaEventRecord(beforeEvent, 0);

	filter_sharedmem<<<dimgrid,dimBlock>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey); // Awful load balance
	cudaDeviceSynchronize();

	// cuda timing events
	cudaEventRecord(afterEvent, 0);
	cudaEventSynchronize(afterEvent);
	cudaEventElapsedTime(&theTime, beforeEvent, afterEvent);

	printf("Time to draw: %f ms\n", theTime);

//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}

// Display images
void Draw()
{
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	if (imagesizey >= imagesizex)
	{ // Not wide - probably square. Original left, result right.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
		glRasterPos2i(0, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE,  pixels);
	}
	else
	{ // Wide image! Original on top, result below.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels );
		glRasterPos2i(-1, 0);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
	}
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

	if (argc > 1)
		image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
	else
		image = readppm((char *)"maskros512.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(2, 2);

// You can save the result to a file like this:
//	writeppm("out.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	return 0;
}
