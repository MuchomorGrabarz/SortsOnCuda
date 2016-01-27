#include <stdio.h>

#include "quicksort.h"

#define middle(a,b,c) (((a > b && b > c) || (c > b && b > c)))

extern "C" {

__global__ void prefixSum(int * input_T, int * prefix_T, int * helper_T) {
    __shared__ int tmp_T[1024], begin, end, pivot;

    if(threadIdx.x == 0) {
        begin = helper_T[6*blockIdx.x];
        end = helper_T[6*blockIdx.x + 1];
        pivot = input_T[helper_T[6*blockIdx.x + 2]];
    }

    __syncthreads();

    for(int i = begin; i <= end; i += 1024) {
        if(i + threadIdx.x > end) return;

        if(input_T[i + threadIdx.x] <= pivot)
            tmp_T[threadIdx.x] = 1;
        else
            tmp_T[threadIdx.x] = 0;


        __syncthreads();
        //printf("(%d : %d : %d)", begin, threadIdx.x, tmp_T[threadIdx.x]);

        __syncthreads();

        if(threadIdx.x == 0 && i != begin) {
            tmp_T[0] += prefix_T[i - 1]; 
            //printf("POSSIBLY STRANGE VAL: %d %d\n", tmp_T[0], i);
        }

        int val, kk = 1;
        while(kk <= 512) {
            __syncthreads();
            if(kk <= threadIdx.x) val = tmp_T[threadIdx.x - kk];
            __syncthreads();
            if(kk <= threadIdx.x) tmp_T[threadIdx.x] += val;
            kk *= 2;
        }

        __syncthreads();

        prefix_T[i + threadIdx.x] = tmp_T[threadIdx.x];

        __syncthreads();

        if(i + threadIdx.x == end) helper_T[6*blockIdx.x + 3] = tmp_T[threadIdx.x];

        __syncthreads();
    }
}

__global__ void rewrite(int * input_T, int * output_T, int * prefix_T, int * helper_T) {
    __shared__ int my_helper[6];

    if(threadIdx.x < 6)
        my_helper[threadIdx.x] = helper_T[blockIdx.x*6 + threadIdx.x];

    if(threadIdx.x == 2) {
        //printf("BLOCK: %d BEGIN: %d END: %d PIVOT: %d PREFIX_ADD: %d SUM: %d\n", blockIdx.x, my_helper[0], my_helper[1], my_helper[2], my_helper[3], my_helper[4]);
        my_helper[2] = input_T[my_helper[2]];
    }

    __syncthreads();

    for(int oldpos = my_helper[0] + threadIdx.x; oldpos <= my_helper[1]; oldpos += 1024) {
        int newpos;
        int prefix = prefix_T[oldpos] + my_helper[3];

        //printf("Old: %d Val: %d Prefix: %d\n", oldpos, input_T[oldpos], prefix);

        if(input_T[oldpos] > my_helper[2])
            newpos = oldpos + my_helper[4] - prefix;
        else
            newpos = my_helper[5] + prefix - 1;

        output_T[newpos] = input_T[oldpos];

        //printf("A na koniec w %d jest %d\n", newpos, output_T[newpos]);
    }
    __syncthreads();
}

struct task {
    int begin;
    int end;
    bool reversed;

    __device__ task(int b, int e, bool r) {
        begin = b;
        end = e;
        reversed = r;
    }

    __device__ task(int b, int e) {
        task(b,e,false);
    }

    __device__ task() {
    }

    __device__ int size() {
        return begin-end+1;
    }
};


struct gpu_task_queue_node {
    gpu_task_queue_node * next;
    task val;
};

struct gpu_task_queue {
    gpu_task_queue_node * begin;
    gpu_task_queue_node * end;
};

__device__ gpu_task_queue * gtqInit() {
    gpu_task_queue * Q = (gpu_task_queue *) malloc(sizeof(gpu_task_queue));
    Q->begin = Q->end = NULL;

    return Q;
}

__device__ bool gtqEmpty(gpu_task_queue * Q) {
    if(Q->begin == NULL) return true;
    else return false;
}

__device__ void gtqPush(gpu_task_queue * Q, task new_task) {
    gpu_task_queue_node * new_end = (gpu_task_queue_node *) malloc(sizeof(gpu_task_queue_node));

    new_end->val = new_task;
    new_end->next = Q->begin;

    if(Q->begin == NULL) {
        Q->begin = new_end;
        Q->end = Q->begin;
    } else {
        Q->begin = new_end;
    }
    Q->begin->val = new_task;
}

__device__ task gtqPop(gpu_task_queue * Q) {
    task result;
    if(Q->begin == NULL) return result;

    result = Q->begin->val;
    gpu_task_queue_node * tmp = Q->begin;

    if(Q->begin->next != NULL) {
        Q->begin = Q->begin->next;
    } else {
        Q->begin = NULL;
        Q->end = NULL;
    }
    free(tmp);

    return result;
}

__device__ int my_rand(int seed) {
    return ((105023 + seed) % 15486277);
}

__global__ void oneBlockMode(int * input_T, int * output_T, int * prefix_T, int * helper_T) {
    __shared__ gpu_task_queue * Q;
    __shared__ int tmp_T[1024];
    __shared__ task current_task;
    __shared__ int sum, pivot, seed;

    __syncthreads();

    if(threadIdx.x == 0) {
        seed = 0;
        Q = gtqInit();
        gtqPush(Q, task(helper_T[blockIdx.x*3], helper_T[blockIdx.x*3 + 1], (bool) helper_T[blockIdx.x*3 + 2]));

        /*printf("BLOCK: %d B: %d E: %d\n", blockIdx.x, helper_T[blockIdx.x*3], helper_T[blockIdx.x*3 + 1]);
        for(int i = helper_T[blockIdx.x*3]; i<=helper_T[blockIdx.x*3 + 1]; i++) {
            if(helper_T[blockIdx.x*3 + 2]) printf("What I have here: %d %d\n", i, input_T[i]);
            else printf("What I have here: %d %d\n", i, output_T[i]);
        }*/
    }

     __syncthreads();

    while(!gtqEmpty(Q)) {
        __syncthreads();

        if(threadIdx.x == 0) {
            current_task = gtqPop(Q);
            //printf("BLOCK: %d B: %d E: %d\n", blockIdx.x, current_task.begin, current_task.end);
        }

        __syncthreads();

        if(current_task.end-current_task.begin < 1024) {
            if(!current_task.reversed && threadIdx.x <= current_task.end-current_task.begin) {
                tmp_T[threadIdx.x] = input_T[current_task.begin + threadIdx.x];
            }
            if(current_task.reversed && threadIdx.x <= current_task.end-current_task.begin) {
                tmp_T[threadIdx.x] = output_T[current_task.begin + threadIdx.x];
            }

            if(threadIdx.x > current_task.end - current_task.begin) {
                tmp_T[threadIdx.x] = 2147483647;
            }

            __syncthreads();

            for(int bSize = 2; bSize <= 1024; bSize *= 2) {
                for(int jump = bSize/2; jump >= 1; jump /= 2) {
                    if(threadIdx.x % (jump*2) < jump && threadIdx.x + jump < 1024  && (
                               ( tmp_T[threadIdx.x] > tmp_T[threadIdx.x + jump] && threadIdx.x % (bSize*2) < bSize ) ||
                               ( tmp_T[threadIdx.x] < tmp_T[threadIdx.x + jump] && threadIdx.x % (bSize*2) >= bSize ))) {
                        tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
                        tmp_T[threadIdx.x + jump] ^= tmp_T[threadIdx.x];
                        tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
                    }
                    __syncthreads();
                }
            }

            if(threadIdx.x <= current_task.end - current_task.begin) output_T[current_task.begin + threadIdx.x] = tmp_T[threadIdx.x];

            continue;
        }

// klasyg
        if(current_task.reversed) {
               input_T = (int *) (((long long) input_T)^((long long) output_T));
               output_T= (int *) (((long long) input_T)^((long long) output_T));
               input_T = (int *) (((long long) input_T)^((long long) output_T));
        }

        if(threadIdx.x == 0) {
            seed = my_rand(seed);
            pivot = input_T[current_task.begin + (seed%(current_task.end - current_task.begin + 1))];
            sum = 0;
        }

        for(int i = current_task.begin; i <= current_task.end; i += 1024) {
            __syncthreads();

            if(i + threadIdx.x <= current_task.end) {
                if(input_T[i + threadIdx.x] <= pivot)
                    tmp_T[threadIdx.x] = 1;
                else
                    tmp_T[threadIdx.x] = 0;
            }

            if(threadIdx.x == 0) {
                tmp_T[0] += sum;
            }

            int kk = 1, val;
            while(kk <= 512) {
                __syncthreads();
                if(kk <= threadIdx.x && i + threadIdx.x <= current_task.end) val = tmp_T[threadIdx.x - kk];
                __syncthreads();
                if(kk <= threadIdx.x && i + threadIdx.x <= current_task.end) tmp_T[threadIdx.x] += val;
                kk *= 2; 
            }

            __syncthreads();

            if(i + threadIdx.x <= current_task.end)
                prefix_T[i + threadIdx.x] = tmp_T[threadIdx.x];

            __syncthreads();

            if(threadIdx.x == 0) {
                if(i + 1023 < current_task.end) {
                    sum = tmp_T[1023];
                } else {
                    sum = tmp_T[current_task.end - i];
                }
            }
        }

        __syncthreads();


        for(int oldpos = current_task.begin + threadIdx.x; oldpos <= current_task.end; oldpos += 1024) {
            int newpos = prefix_T[oldpos];

            if(input_T[oldpos] <= pivot) {
                newpos = current_task.begin + newpos - 1;
            } else {
                newpos = oldpos + sum - newpos;
            }

            output_T[newpos] = input_T[oldpos];
        }

        __syncthreads();

        if(threadIdx.x == 0) {
            if(current_task.end - current_task.begin < sum || sum <= 0) {
                gtqPush(Q, current_task);
            } else {
                gtqPush(Q, task(current_task.begin, current_task.begin+sum-1, !current_task.reversed));
                gtqPush(Q, task(current_task.begin+sum, current_task.end, !current_task.reversed));
            }
        }

        if(current_task.reversed) {
               input_T = (int *) (((long long) input_T)^((long long) output_T));
               output_T= (int *) (((long long) input_T)^((long long) output_T));
               input_T = (int *) (((long long) input_T)^((long long) output_T));
        }

        __syncthreads();
    }

//    printf("BLOCK: %d ended\n", blockIdx.x);
}

}
