#include <cstdio>

#define getPos(a,k) (((a)>>(k-1))&1)

extern "C" {

__global__ void prefixSum(int * input_T, int * prefix_T, int * prefix_helper_T, int n, int k, int blockPower) {
    __shared__ int tmp_T[1024];

    for(int i = 0; i<blockPower; i++) {
        if(threadIdx.x + 1024*blockIdx.x + i*1024*gridDim.x >= n) return;

        tmp_T[threadIdx.x] = input_T[threadIdx.x + 1024*blockIdx.x + i*1024*gridDim.x];
        tmp_T[threadIdx.x] = getPos(tmp_T[threadIdx.x],k);

        int val,kk = 1;
        while(kk <= 512) {
            __syncthreads();
            if(kk <= threadIdx.x) val = tmp_T[threadIdx.x - kk];
            __syncthreads();
            if(kk <= threadIdx.x) tmp_T[threadIdx.x] += val;
            kk *= 2;
        }

        __syncthreads();

        prefix_T[threadIdx.x + 1024*blockIdx.x + i*1024*gridDim.x] = tmp_T[threadIdx.x];

        if(threadIdx.x == 1023 || threadIdx.x + 1024*blockIdx.x + i*1024*gridDim.x == n-1) prefix_helper_T[i*gridDim.x + blockIdx.x + 1] = tmp_T[threadIdx.x];
    }
}

__global__ void replace(int * input_T, int * output_T, int * prefix_T, int * prefix_helper_T, int n, int k, int blockPower) {
    for(int i = 0; i<blockPower; i++) {
        int oldpos = threadIdx.x + 1024*blockIdx.x + i*1024*gridDim.x;
        if(oldpos >= n) return ;


        int newpos = prefix_T[oldpos] + prefix_helper_T[blockIdx.x + i*gridDim.x];

        if(getPos(input_T[oldpos],k) == 0) {
            newpos = oldpos - newpos;
        } else {
            newpos = prefix_helper_T[(n+1023)/1024] + newpos - 1;
        }

        output_T[newpos] = input_T[oldpos];
    }

}

}
