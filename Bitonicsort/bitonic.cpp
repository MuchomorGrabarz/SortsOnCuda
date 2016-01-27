#include <stdio.h>
#include <stdlib.h>

#include "bitonic.h"
#include "cuda.h"

#define PROCESSORS 8

#define min(a,b) (a<b?a:b)

using namespace std;

int * bitonic(int * tab, int n) {

	//Powiedzmy sobie szczerze - tego do wykrzykników nie ma co ruszać
	cuInit(0); 

	CUdevice cuDevice;
	CUresult res = cuDeviceGet(&cuDevice, 0);
	if (res != CUDA_SUCCESS){
		printf("cannot acquire device 0\n");
		exit(1);
	}

	CUcontext cuContext;
	res = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST | CU_CTX_SCHED_BLOCKING_SYNC, cuDevice);
	if (res != CUDA_SUCCESS){
		printf("cannot create context\n");
		exit(1);
	}

	// !!! Tutaj 

	CUmodule cuModule = (CUmodule)0;
	res = cuModuleLoad(&cuModule, "bitonic.ptx");
	if (res != CUDA_SUCCESS) {
		printf("cannot load module: %d\n", res);  
		exit(1); 
	}

	/// Możemy se co najwyżej podmienić nazwę pliku
	
	CUfunction init;
	res = cuModuleGetFunction(&init, cuModule, "init");
	if (res != CUDA_SUCCESS){
		printf("cannot acquire kernel handle\n");
		exit(1);
	}
	
	CUfunction oneMove;
	res = cuModuleGetFunction(&oneMove, cuModule, "oneMove");
	if (res != CUDA_SUCCESS){
		printf("cannot acquire kernel handle\n");
		exit(1);
	}

	CUfunction oneReduction;
	res = cuModuleGetFunction(&oneReduction, cuModule, "oneReduction");
	if (res != CUDA_SUCCESS){
		printf("cannot acquire kernel handle\n");
		exit(1);
	}

	CUfunction oneBlock;
	res = cuModuleGetFunction(&oneBlock, cuModule, "oneBlock");
	if (res != CUDA_SUCCESS){
		printf("cannot acquire kernel handle\n");
		exit(1);
	}

	int new_n = 1;
	while(new_n < n) new_n *= 2;

	CUdeviceptr gpu_tab;
	res = cuMemAlloc(&gpu_tab, new_n*sizeof(int));
	if (res != CUDA_SUCCESS){
		printf("cannot alloc the memory\n");
		exit(1);
	}

	int len = new_n/PROCESSORS;

	void * args1[] = {&gpu_tab, &len};
	res = cuLaunchKernel(init, PROCESSORS, 1, 1, 1024, 1, 1, 0, 0, args1, 0);
	cuCtxSynchronize();
	if (res != CUDA_SUCCESS){
		printf("cannot run first kernel properely\n");
		exit(1);
	}
	cuCtxSynchronize();

	res = cuMemcpyHtoD(gpu_tab, (void *) tab, n*sizeof(int));

	// tutaj odpalamy 8 wątków

	void * args2[] = {&gpu_tab, &len};
	res = cuLaunchKernel(oneBlock, PROCESSORS, 1, 1, 1024, 1, 1, 0, 0, args2, 0);
	cuCtxSynchronize();
	if (res != CUDA_SUCCESS){
		printf("cannot run second kernel properely: %d\n", res);
		exit(1);
	}
    int * tmpRes = (int *) malloc(new_n*sizeof(int));
    cuMemcpyDtoH((void*)tmpRes, gpu_tab, new_n*sizeof(int));
    for(int i = 0; i < 16; i += 2) {
        for(int k = i*len/2 + 1; k < (i+1)*len/2; k++) {
            if(tmpRes[k] < tmpRes[k-1]) printf("FAILS %d : %d PREV: %d\n", k, tmpRes[k], tmpRes[k-1]);
        }
        for(int k = (i+1)*len/2 + 1; k < (i+2)*len/2; k++) {
            if(tmpRes[k] > tmpRes[k-1]) printf("FAILS %d : %d PREV: %d\n", k, tmpRes[k], tmpRes[k-1]);
        }
    }

	int mod = 2;

    void * args3[] = {&gpu_tab, &len, &mod};
    res = cuLaunchKernel(oneReduction, PROCESSORS, 1, 1, 1024, 1, 1, 0, 0, args3, 0);
    cuCtxSynchronize();

    cuMemcpyDtoH((void*)tmpRes, gpu_tab, new_n*sizeof(int));

    printf("Tu jeszcze ok?\n");

    for(int i = 0; i < 8; i += 2) {
        for(int k = i*len + 1; k < (i+1)*len; k++) {
            if(tmpRes[k] < tmpRes[k-1]) printf("FAILS %d : %d PREV: %d\n", k, tmpRes[k], tmpRes[k-1]);
        }
        for(int k = (i+1)*len + 1; k < (i+2)*len; k++) {
            if(tmpRes[k] > tmpRes[k-1]) printf("FAILS %d : %d PREV: %d\n", k, tmpRes[k], tmpRes[k-1]);
        }
    }

    if (res != CUDA_SUCCESS){
        printf("cannot run third kernel properely\n");
        exit(1);
    }

	for( int k = len; k < new_n; k *= 2 ) {
		for( int dist = k; dist >= len; dist /= 2 ) {
		    int blockPower = new_n/(1024*PROCESSORS);
		    int blocksPerTask = PROCESSORS/(new_n/(dist*2));
            int ascDescPeriod =  PROCESSORS/(new_n/(k*2));

			void * args4[] = {&gpu_tab, &dist, &blockPower, &blocksPerTask, &ascDescPeriod};
			res = cuLaunchKernel(oneMove, PROCESSORS, 1, 1, 1024, 1, 1, 0, 0, args4, 0);
			if (res != CUDA_SUCCESS){
				printf("cannot run fourth kernel properely\n");
				exit(1);
			}
			cuCtxSynchronize();
		}

		mod = 4*(k/len);

		void * args5[] = {&gpu_tab, &len, &mod};
		res = cuLaunchKernel(oneReduction, PROCESSORS, 1, 1, 1024, 1, 1, 0, 0, args5, 0);
		cuCtxSynchronize();

		if (res != CUDA_SUCCESS){
			printf("cannot run fourth kernel properely\n");
			exit(1);
		}
	}

	int * result = (int *) malloc(n*sizeof(int));
	res = cuMemcpyDtoH((void *) result, gpu_tab, n*sizeof(int));

	cuMemFree(gpu_tab);
	cuCtxDestroy(cuContext);

	return result;
}
