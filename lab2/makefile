edit: 
	nvcc filter.cu  -c -arch=sm_30 -o filter.o
	g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter

	nvcc filter_sharedmem.cu -c -arch=sm_30 -o filter_sharedmem.o
	g++ filter_sharedmem.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter_sharedmem

	nvcc filter_separable.cu -c -arch=sm_30 -o filter_separable.o
	g++ filter_separable.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter_separable
	
	nvcc filter_gauss_sep.cu -c -arch=sm_30 -o filter_gauss_sep.o
	g++ filter_gauss_sep.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter_gauss_sep

	nvcc filter_median.cu -c -arch=sm_30 -o filter_median.o
	g++ filter_median.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter_median