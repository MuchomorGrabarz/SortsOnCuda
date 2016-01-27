#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <queue>
#include <vector>

#include "radixsort.h"
#include "cuda.h"

#define min(a,b) (a<b?a:b)

using namespace std;

int * radixsort(int * T, int n) {

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
	res = cuModuleLoad(&cuModule, "radixsort.ptx");
	if (res != CUDA_SUCCESS) {
		printf("cannot load module: %d\n", res);  
		exit(1); 
	}

	/// Możemy se co najwyżej podmienić nazwę pliku

	// liczenie liczby bloków
	
	int blockNum;
	if(n < 1024) blockNum = (n+1023)/1024;
    else blockNum = 1024;

    int blockPower = (n+blockNum*1024-1)/(blockNum*1024);

	// tu ciskamy alokację
	
	int * result_T = (int *) malloc(sizeof(int)*n);

    CUdeviceptr gpu_T1, gpu_T2, gpu_prefix_T;

    cuMemAlloc(&gpu_T1, n*sizeof(int));
    cuMemAlloc(&gpu_T2, n*sizeof(int));
	cuMemAlloc(&gpu_prefix_T, n*sizeof(int));

	cuMemcpyHtoD(gpu_T1, T, n*sizeof(int));

	void * prefix_helper_T;
	cuMemAllocHost(&prefix_helper_T, ((n+1023)/1024)*sizeof(int));

	// odpalamy kernela
	
    CUfunction prefixSum;
    res = cuModuleGetFunction(&prefixSum, cuModule, "prefixSum");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

    CUfunction replace;
    res = cuModuleGetFunction(&replace, cuModule, "replace");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

    bool mark = false;

//    printf("%d\n", blockNum);

    for(int k = 1; k<=31; k++) {
        CUdeviceptr gpu_local_T1, gpu_local_T2;

        if(mark) {
            gpu_local_T1 = gpu_T2;
            gpu_local_T2 = gpu_T1;
        } else {
            gpu_local_T1 = gpu_T1;
            gpu_local_T2 = gpu_T2;
        }

        mark = not mark;

        void* args[] = {&gpu_local_T1, &gpu_prefix_T, &prefix_helper_T, &n, &k, &blockPower};
//        printf("Block power : %d\n", blockPower);

        res = cuLaunchKernel(prefixSum, blockNum, 1, 1, 1024, 1, 1, 0, 0, args, 0);

        cuCtxSynchronize();

        if (res != CUDA_SUCCESS){
            printf("cannot run first kernel: %d\n", res);
            exit(1);
        }

        ((int *) prefix_helper_T)[0] = 0;
        for(int i = 1; i<=(n+1023)/1024; i++) {
            ((int *) prefix_helper_T)[i] += ((int *) prefix_helper_T)[i-1];
        }
        ((int *) prefix_helper_T)[(n+1023)/1024] = n - ((int *) prefix_helper_T)[(n+1023)/1024];

        void* args2[] = {&gpu_local_T1, &gpu_local_T2, &gpu_prefix_T, &prefix_helper_T, &n, &k, &blockPower};

        res = cuLaunchKernel(replace, blockNum, 1, 1, 1024, 1, 1, 0, 0, args2, 0);

        cuCtxSynchronize();

        if (res != CUDA_SUCCESS){
            printf("cannot run second kernel: %d\n", res);
            exit(1);
        }
    }

	// tu ciskamy kopiowanko z pamięci GPU
	
    res = cuMemcpyDtoH((void *) result_T,gpu_T2,n*sizeof(int));

	cuMemFree(gpu_T1);
	cuMemFree(gpu_T2);
	cuMemFree(gpu_prefix_T);
	cuMemFreeHost(prefix_helper_T);

	cuCtxDestroy(cuContext);

	return result_T;
}
