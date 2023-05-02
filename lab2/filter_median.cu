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

// Use these for setting the filter kernal size
#define filterSizeX 8
#define filterSizeY 8

const int blockSize = 32; // #threads per block

// Use this for setting the image file name
#define FileName "maskros-noisy.ppm" // files names to choose from baboon1.ppm, maskros512.ppm, maskros-noisy.ppm, img1.ppm, img1-noisy.ppm


// This is the original code -----------
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

__global__ void filter_median(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{
	// creating a tile scaling 
	const int tile = blockDim.x - 2*maxKernelSizeX;
	
    // map from blockIdx to pixel position
    int x = blockIdx.x * tile + threadIdx.x-kernelsizex;
    int y = blockIdx.y * tile + threadIdx.y-kernelsizey;
    
	// Use max and min to avoid branching!
    x = min(max(x, 0), imagesizex-1);
    y = min(max(y, 0), imagesizey-1);

	// shared memory initialization
	__shared__ unsigned char imagebuffer[blockSize*blockSize*3];

    imagebuffer[(threadIdx.x + blockDim.x*threadIdx.y)*3+0] = image[(imagesizex*y + x)*3+0];
    imagebuffer[(threadIdx.x + blockDim.x*threadIdx.y)*3+1] = image[(imagesizex*y + x)*3+1];
    imagebuffer[(threadIdx.x + blockDim.x*threadIdx.y)*3+2] = image[(imagesizex*y + x)*3+2];

    __syncthreads();
    // Make local copy for each thread

    unsigned char r[(maxKernelSizeX*2+1)*(maxKernelSizeY*2+1)];
    unsigned char g[(maxKernelSizeX*2+1)*(maxKernelSizeY*2+1)];
    unsigned char b[(maxKernelSizeX*2+1)*(maxKernelSizeY*2+1)];
    
    unsigned char temp;
    int count = 0;

    int dx, dy;

    if((threadIdx.x >= (kernelsizex)) && (threadIdx.x < ((blockDim.x-(kernelsizex)))) && (threadIdx.y >= kernelsizey) && (threadIdx.y < (blockDim.y-kernelsizey))) {
     	 for(dy=-kernelsizey;dy<=kernelsizey;dy++)
       {
       		for(dx=-kernelsizex;dx<=kernelsizex;dx++)
       		{
             r[count] = imagebuffer[((threadIdx.x + blockDim.x*threadIdx.y)+(dy*blockDim.x)+dx)*3+0];
             g[count] = imagebuffer[((threadIdx.x + blockDim.x*threadIdx.y)+(dy*blockDim.x)+dx)*3+1];
             b[count] = imagebuffer[((threadIdx.x + blockDim.x*threadIdx.y)+(dy*blockDim.x)+dx)*3+2];
             count++;
       		}
      }

      // Bubblesort to determine the median vlaue
      for (int i = 0; i < ((2*kernelsizex+1)*(2*kernelsizey+1))-1; i++){
        for (int j = 0; j < ((2*kernelsizex+1)*(2*kernelsizey+1))-i-1; j++) {
          if (r[j] > r[j+1]) {
             temp = r[j];
             r[j] = r[j+1];
             r[j+1] = temp;
          }
          if (g[j] > g[j+1]) {
             temp = g[j];
             g[j] = g[j+1];
             g[j+1] = temp;
          }
          if (b[j] > b[j+1]) {
             temp = b[j];
             b[j] = b[j+1];
             b[j+1] = temp;
          }
        }
      }
      unsigned int median_val = ((2*kernelsizex+1)*(2*kernelsizey+1)-1)/2;


      out[(imagesizex*y + x)*3+0] = r[median_val];
      out[(imagesizex*y + x)*3+1] = g[median_val];
      out[(imagesizex*y + x)*3+2] = b[median_val];
    }
}

// Global variables for image data
unsigned char *image, *pixels, *dev_bitmap, *dev_input, *temp_bitmap;
unsigned int imagesizey, imagesizex; // Image size
// int *stencil_d; // gaussian filter gain

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

    // cudaMalloc( (void**)&temp_bitmap, imagesizex*imagesizey*3); // temprary memory

	// int stencil_h[5] = {1,4,6,4,1};
    // cudaMalloc((void**)&stencil_d, 5*sizeof(int)); 
    // cudaMemcpy(stencil_d, stencil_h, 5*sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimBlock(blockSize,blockSize);
	dim3 dimgrid(imagesizex/maxKernelSizeX,imagesizey/maxKernelSizeY);
    // dim3 dimgrid((imagesizex/(blockSize-kernelsizex*2)+1), (imagesizex/(blockSize-kernelsizey*2)+1));

	// creating cuda events for timinfilterSizeXg
	cudaEvent_t beforeEvent;
	cudaEvent_t afterEvent;
	float theTime;
	cudaEventCreate(&beforeEvent);
	cudaEventCreate(&afterEvent);
	cudaEventRecord(beforeEvent, 0);

	// Task 0 (default)
	filter_median<<<dimgrid,dimBlock>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey); // Awful load balance
    cudaDeviceSynchronize();

	// cuda timing events
	cudaEventRecord(afterEvent, 0);
	cudaEventSynchronize(afterEvent);
	cudaEventElapsedTime(&theTime, beforeEvent, afterEvent);

	printf("Time to draw: %f us\n", theTime*1000);

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
		image = readppm((char *)FileName, (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(filterSizeX, filterSizeY);

// You can save the result to a file like this:
//	writeppm("out.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	return 0;
}
