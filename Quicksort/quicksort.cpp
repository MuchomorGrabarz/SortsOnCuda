#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include <queue>
#include <vector>
#include <map>

#include "quicksort.h"
#include "cuda.h"

#define min(a,b) (a<b?a:b)
#define STANDARD_BLOCK_NUM 124

using namespace std;

struct task {
    long long begin;
    long long end;

    int firstBlock;
    int lastBlock;

    bool reversed;

    task(long long b, long long e, bool r) {
        begin = b;
        end = e;
        reversed = r;
    }

    task(long long b, long long e) {
        task(b,e,false);
    }

    task() {
        task(0,0);
    }

    long long size() {
        return end-begin+1L;
    }
};

int * quicksort(int * T, int n) {

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
	res = cuModuleLoad(&cuModule, "quicksort.ptx");
	if (res != CUDA_SUCCESS) {
		printf("cannot load module: %d\n", res);  
		exit(1); 
	}


	// szykujemy kernele
	
    CUfunction prefixSum;
    res = cuModuleGetFunction(&prefixSum, cuModule, "prefixSum");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

    CUfunction rewrite;
    res = cuModuleGetFunction(&rewrite, cuModule, "rewrite");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

    CUfunction oneBlockMode;
    res = cuModuleGetFunction(&oneBlockMode, cuModule, "oneBlockMode");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

	// przygotowujemy tablicę
	
    CUdeviceptr gpu_T1, gpu_T2, gpu_prefix_T;
    void * helper_T;

    int BLOCK_NUM = min(STANDARD_BLOCK_NUM, (1023 + n)/1024);

    cuMemAlloc(&gpu_T1, n*sizeof(int));
    cuMemAlloc(&gpu_T2, n*sizeof(int));
	cuMemAlloc(&gpu_prefix_T, n*sizeof(int));
	cuMemAllocHost(&helper_T, 40*BLOCK_NUM*sizeof(int));

	res = cuMemcpyHtoD(gpu_T1, T, n*sizeof(int));

	if(res != CUDA_SUCCESS) {
        printf("Another error.\n");
        exit(1);
    }

    // przygotowujemy zmienne

    bool marked = false;

    queue<task> waitingTasks, tasksInProgress;
    vector<task> readyTasks;

    waitingTasks.push(task(0, n-1, false));

    int numberOfTasks = 0;
    int expectedSize = 1 + n/BLOCK_NUM;
    long long lenOfTheRest = (long long) n;

    // dzielimy tablicę na strawne kawałki

    while(!waitingTasks.empty() && waitingTasks.size() < (long long) BLOCK_NUM) {
        long long numberOfBlocks = 0L;

        // prepare helper
        
        while(!waitingTasks.empty()) {
            task tmp = waitingTasks.front();
            waitingTasks.pop();

            // get random pivot

            srand(time(NULL));
            int pivot = tmp.begin + rand()%(tmp.end - tmp.begin + 1);

            long long howManyBlocks = ((((long long) BLOCK_NUM)*tmp.end - tmp.begin + 1L) + lenOfTheRest - 1L)/lenOfTheRest; // uzupełnić
            //printf("HOW MANY BLOCKS: %lld\n", howManyBlocks);
            
            for(long long i = 0; i < howManyBlocks - 1L; i++) {
                ((int*) helper_T)[6*(numberOfBlocks + i)] = (int) tmp.begin + (i*(tmp.end-tmp.begin))/howManyBlocks;
                ((int*) helper_T)[6*(numberOfBlocks + i) + 1] = (int) tmp.begin + ((i+1)*(tmp.end-tmp.begin))/howManyBlocks - 1;
                ((int*) helper_T)[6*(numberOfBlocks + i) + 2] = (int) pivot;

                //printf("block %d : begin %d  end %d\n", numberOfBlocks+i, ((int*) helper_T)[6*(numberOfBlocks + i)], ((int*) helper_T)[6*(numberOfBlocks + i) + 1]);
            }

            ((int*) helper_T)[6*(numberOfBlocks + howManyBlocks - 1)] = (int) tmp.begin + ((howManyBlocks-1)*(tmp.end-tmp.begin))/howManyBlocks;
            ((int*) helper_T)[6*(numberOfBlocks + howManyBlocks - 1) + 1] = (int) tmp.end;
            ((int*) helper_T)[6*(numberOfBlocks + howManyBlocks - 1) + 2] = (int) pivot;
            //printf("block %d : begin %d  end %d\n", numberOfBlocks+howManyBlocks-1, ((int*) helper_T)[6*(numberOfBlocks + howManyBlocks - 1)], ((int*) helper_T)[6*(numberOfBlocks + howManyBlocks - 1) + 1]);


            tmp.firstBlock = numberOfBlocks;
            tmp.lastBlock = numberOfBlocks + howManyBlocks - 1L;

            numberOfBlocks += howManyBlocks;
            numberOfTasks++;

            tasksInProgress.push(tmp);

            //printf("Start info: blocks: %d start: %d end: %d pivot: %d\n", howManyBlocks, tmp.begin, tmp.end, pivot);
            //fflush(NULL);
        }

        CUdeviceptr gpu_local_T1, gpu_local_T2;

        if(marked) {
            gpu_local_T1 = gpu_T2;
            gpu_local_T2 = gpu_T1;
        } else {
            gpu_local_T1 = gpu_T1;
            gpu_local_T2 = gpu_T2;
        }

        void* args[] = {&gpu_local_T1, &gpu_prefix_T, &helper_T};
        res = cuLaunchKernel(prefixSum, numberOfBlocks, 1, 1, 1024, 1, 1, 0, 0, args, 0);
        cuCtxSynchronize();


        if (res != CUDA_SUCCESS){
            printf("cannot run first kernel: %d\n", res);
            exit(1);
        }

        lenOfTheRest = 0;

        // tutaj użyj wyników -- to też jest chyba istotne
        while(!tasksInProgress.empty()) {
            task tmp = tasksInProgress.front();
            tasksInProgress.pop();

            for(int i = tmp.firstBlock + 1; i <= tmp.lastBlock; i++) {
                ((int*) helper_T)[i*6 + 3] += ((int*) helper_T)[(i-1)*6 + 3]; 
            }

            int div = ((int*) helper_T)[tmp.lastBlock*6 + 3];

            for(int i = tmp.firstBlock; i <= tmp.lastBlock; i++) {
                ((int*) helper_T)[i*6 + 4] = div; 
                ((int*) helper_T)[i*6 + 5] = tmp.begin;
            }

            for(int i = tmp.lastBlock; i > tmp.firstBlock; i--) {
                ((int*) helper_T)[i*6 + 3] = ((int*) helper_T)[(i-1)*6 + 3]; 
            }
            ((int*) helper_T)[tmp.firstBlock*6 + 3] = 0;
            
            //printf("Proseccing info: begin: %d, end: %d, firstBlock: %d, lastBlock: %d, div: %d\n", tmp.begin, tmp.end, tmp.firstBlock, tmp.lastBlock, div);
            //printf("First begin: %d first end: %d\n", tmp.begin, tmp.begin+div-1);
            //printf("Second begin: %d second end: %d\n", tmp.begin + div, tmp.end);
            //fflush(NULL);

            task firstTask(tmp.begin, tmp.begin + div - 1, not marked); // z tym marked to jeszcze nie wiadomo
            if(firstTask.begin <= firstTask.end) {
                if(firstTask.size() > expectedSize && firstTask.size() > 1024) {
                    lenOfTheRest += firstTask.size();
                    waitingTasks.push(firstTask);
                } else {
                    readyTasks.push_back(firstTask);
                }
            }

            task secondTask(tmp.begin + div, tmp.end, not marked); // as well
            if(secondTask.begin <= secondTask.end) {
                if(secondTask.size() > expectedSize && secondTask.size() > 1024) {
                    lenOfTheRest += secondTask.size();
                    waitingTasks.push(secondTask);
                } else {
                    readyTasks.push_back(secondTask);
                }
            }
        }


        cuCtxSynchronize();
        void* args2[] = {&gpu_local_T1, &gpu_local_T2, &gpu_prefix_T, &helper_T};
        res = cuLaunchKernel(rewrite, numberOfBlocks, 1, 1, 1024, 1, 1, 0, 0, args2, 0);
        cuCtxSynchronize();

        if (res != CUDA_SUCCESS){
            printf("cannot run second kernel: %d\n", res);
            exit(1);
        }

        marked = not marked;

    }


    // szykujemy się do odpalenia ostatecznego sortowania

    int numberOfBlocks = 0;

    while(!waitingTasks.empty()) {
        readyTasks.push_back(waitingTasks.front());
        waitingTasks.pop();
    }

    for(int i = 0; i<readyTasks.size(); i++) {
 //      printf("Task begin: %d und end: %d\n", readyTasks[i].begin, readyTasks[i].end);
       ((int*) helper_T)[3*i] = readyTasks[i].begin;
       ((int*) helper_T)[3*i + 1] = readyTasks[i].end;
       ((int*) helper_T)[3*i + 2] = (int) readyTasks[i].reversed;
    }

    // odpalamy ostateczne sortowanie

    cuCtxSynchronize();
    void* args[] = {&gpu_T1, &gpu_T2, &gpu_prefix_T, &helper_T};
    res = cuLaunchKernel(oneBlockMode, readyTasks.size(), 1, 1, 1024, 1, 1, 0, 0, args, 0);
    cuCtxSynchronize();

    if (res != CUDA_SUCCESS){
        printf("cannot run third kernel: %d\n", res);
        exit(1);
    }

    // kopiujemy i zwalniamy pamięć

    int * result_T = (int *) malloc(n*sizeof(int));
    res = cuMemcpyDtoH((void *) result_T, gpu_T2, n*sizeof(int));
    if (res != CUDA_SUCCESS){
        printf("cannot run fourth kernel: %d\n", res);
        exit(1);
    }
    cuCtxSynchronize();

 //   for(int i = 0; i<n; i++)
//        printf("A taki jest rezultat w i: %d\n", (result_T)[i]);

	cuMemFree(gpu_T1);
	cuMemFree(gpu_T2);
	cuMemFree(gpu_prefix_T);
	cuMemFreeHost(helper_T);

	cuCtxDestroy(cuContext);

	return result_T;
}
