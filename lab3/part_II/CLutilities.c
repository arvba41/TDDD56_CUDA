// CL utilities by Ingemar
// readFile is just grabbed from my shader utilities.
// 2017: Extended with initOpenCL, closeOpenCL and compileKernel.
// Warning: These are packaged for your convenience but are not
// made for flexibility. You may want to edit them for heavy use.
// Simplifies the main program a lot to make it suitable for course work.
// 2021: Corrected the declaration of the three global variables below
// so that they are not dependent on compiler settings.

#include "CLutilities.h"

// global variables needed after initialization
cl_context cxGPUContext;
cl_command_queue commandQueue;

// Convenient global (from clGetDeviceInfo)
int gMaxThreadsPerWG;

static cl_device_id device;

char* readFile(const char * filename)
{
	char * data;
	FILE *theFile;
	char c;
	long howMuch;
	
	// Get file length
	theFile = fopen(filename, "rb");
	if (theFile == NULL)
	{
		printf("%s not found\n", filename);
		return NULL;
	}
	howMuch = 0;
	c = 0;
	while (c != EOF)
	{
		c = getc(theFile);
		howMuch++;
	}
	fclose(theFile);

	printf("%ld bytes\n", howMuch);
	
	// Read it again
	data = (char *)malloc(howMuch);
	theFile = fopen(filename, "rb");
	fread(data, howMuch-1,1,theFile);
	fclose(theFile);
	data[howMuch-1] = 0;

//	printf("%s\n-----\n", data);
	printf("%s loaded from disk\n", filename);

	return data;
}

void printCLError(cl_int ciErrNum, int location)
{	
    if (ciErrNum != CL_SUCCESS)
    {
      switch (location)
      {
        case 0:
          printf("Error @ clGetPlatformIDs: ");
          break;
        case 1:
          printf("Error @ clGetDeviceIDs: ");
          break;
        case 2:
          printf("Error @ clCreateContext: ");
          break;
        case 3:
          printf("Error @ clGetDeviceInfo: ");
          break;
        case 4:
          printf("Error @ clCreateCommandQueue: ");
          break;
        case 5:
          printf("Error @ clCreateProgramWithSource: ");
          break;
        case 6:
          printf("Error @ clCreateKernel: ");
          break;
        case 7:
          printf("Error @ clCreateBuffer: ");
          break;
        case 8:
          printf("Error @ clSetKernelArg: ");
          break;
        case 9:
          printf("Error @ clEnqueueNDRangeKernel: ");
          break;
        case 10:
          printf("Error @ clWaitForEvents: ");
          break;
        case 11:
          printf("Error @ clEnqueueReadBuffer: ");
          break;
        case 12:
          printf("Error @ clBuildProgram: ");
          break;
        default:
          printf("Error @ unknown location: ");
          break;
      }
      switch (ciErrNum)
      {
        case CL_INVALID_PROGRAM_EXECUTABLE:
          printf("CL_INVALID_PROGRAM_EXECUTABLE\n");
          break;
        case CL_INVALID_COMMAND_QUEUE:
          printf("CL_INVALID_COMMAND_QUEUE\n");
          break;
        case CL_INVALID_KERNEL:
          printf("CL_INVALID_KERNEL\n");
          break;
        case CL_INVALID_CONTEXT:
          printf("CL_INVALID_CONTEXT\n");
          break;
        case CL_INVALID_KERNEL_ARGS:
          printf("CL_INVALID_KERNEL_ARGS\n");
          break;
        case CL_INVALID_WORK_DIMENSION:
          printf("CL_INVALID_WORK_DIMENSION\n");
          break;
        case CL_INVALID_WORK_GROUP_SIZE:
          printf("CL_INVALID_WORK_GROUP_SIZE\n");
          break;
        case CL_INVALID_WORK_ITEM_SIZE:
          printf("CL_INVALID_WORK_ITEM_SIZE\n");
          break;
        case CL_INVALID_GLOBAL_OFFSET:
          printf("CL_INVALID_GLOBAL_OFFSET\n");
          break;
        case CL_OUT_OF_RESOURCES:
          printf("CL_OUT_OF_RESOURCES\n");
          break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
          printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n");
          break;
        case CL_INVALID_EVENT_WAIT_LIST:
          printf("CL_INVALID_EVENT_WAIT_LIST\n");
          break;
        case CL_OUT_OF_HOST_MEMORY:
          printf("CL_OUT_OF_HOST_MEMORY\n");
          break;
        case CL_INVALID_MEM_OBJECT:
          printf("CL_INVALID_MEM_OBJECT\n");
          break;
        case CL_INVALID_VALUE:
          printf("CL_INVALID_VALUE\n");
          break;
        case CL_INVALID_PROGRAM:
          printf("CL_INVALID_PROGRAM\n");
          break;
        case CL_INVALID_KERNEL_DEFINITION:
          printf("CL_INVALID_KERNEL_DEFINITION\n");
          break;
        case CL_INVALID_PLATFORM:
          printf("CL_INVALID_PLATFORM\n");
          break;
        case CL_INVALID_DEVICE_TYPE:
          printf("CL_INVALID_DEVICE_TYPE\n");
          break;
        case CL_DEVICE_NOT_FOUND:
          printf("CL_DEVICE_NOT_FOUND\n");
          break;
        
        default:
          printf("Error: Unknown error\n");
          break;
      }
      exit(1);
    }
}



int initOpenCL()
{
  cl_int ciErrNum = CL_SUCCESS;
  cl_platform_id platform;
  unsigned int no_plat;
  size_t noWG;
  
  // Assume that there is only one platform.
  ciErrNum =  clGetPlatformIDs(1,&platform,&no_plat);
  printCLError(ciErrNum,0);

  //get the device
  ciErrNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  printCLError(ciErrNum,1);
  
  // create the OpenCL context on the device
  cxGPUContext = clCreateContext(0, 1, &device, NULL, NULL, &ciErrNum);
  printCLError(ciErrNum,2);
  
  // Check out what we got
  ciErrNum = clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&noWG,NULL);
  printCLError(ciErrNum,3);
  printf("maximum work group size: %d\n", (int)noWG);
  gMaxThreadsPerWG = (int)noWG;

//	size_t workitem_size[3];
//	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
//	printf("  CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%u / %u / %u \n", workitem_size[0], workitem_size[1], workitem_size[2]);
  
  // create command queue
  commandQueue = clCreateCommandQueueWithProperties(cxGPUContext, device, 0, &ciErrNum);
  printCLError(ciErrNum,4);

  return 0;
}

cl_kernel compileKernel(char *filename, char *kernelName)
{
  cl_int ciErrNum = CL_SUCCESS;
  cl_kernel kernel;
  static cl_program program;
  size_t kernelLength;
  char *source;

  // Read the kernel file
  source = readFile(filename);
  kernelLength = strlen(source);
  
  // create the program
  program = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&source, 
                                                    &kernelLength, &ciErrNum);
  printCLError(ciErrNum,5);
    
  // build the program
  ciErrNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (ciErrNum != CL_SUCCESS)
  {
    printf("%s build failed!\n", filename);
    // write out the build log, then exit
    char cBuildLog[10240];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
                          sizeof(cBuildLog), cBuildLog, NULL );
    printf("\nBuild Log:\n%s\n\n", (char *)&cBuildLog);
    return NULL;
  }
  
  kernel = clCreateKernel(program, kernelName, &ciErrNum);
  printCLError(ciErrNum,6);
  
  //Discard temp storage
  free(source);
  clReleaseProgram(program);

  printf("%s built\n", filename);
  
  return kernel;
}

void closeOpenCL()
{
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (cxGPUContext) clReleaseContext(cxGPUContext);
}