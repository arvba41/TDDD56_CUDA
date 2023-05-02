#ifndef __INGEMARS_CL_UTILITIES_
#define __INGEMARS_CL_UTILITIES_

// Change 2021: "defined version to 220 (you may want to change this)
// Changed clCreateCommandQueue to clCreateCommandQueueWithProperties
// You might need to change this back on older versions.
// 2021: Corrected the declaration of the three global variables below
// so that they are not dependent on compiler settings.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef CL_TARGET_OPENCL_VERSION
  #define CL_TARGET_OPENCL_VERSION 220
#endif
#ifdef __APPLE__
  #include <OpenCL/opencl.h>
  // At least my Mac needs this:
  #define clCreateCommandQueueWithProperties clCreateCommandQueue
#else
  #include <CL/cl.h>
#endif

// global variables needed after initialization
extern cl_context cxGPUContext;
extern cl_command_queue commandQueue;

// Convenient global (from clGetDeviceInfo)
extern int gMaxThreadsPerWG;

char* readFile(const char * filename);
void printCLError(cl_int ciErrNum, int location);

int initOpenCL();
cl_kernel compileKernel(char *filename, char *kernelName);
void closeOpenCL();

#endif
