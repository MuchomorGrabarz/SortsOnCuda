#include <stdio.h>
#include <stdlib.h>

#define INF 2147483647

extern "C" {

__global__ void init(int * tab, int len) {
	for(int i = threadIdx.x + len*blockIdx.x; i < len*blockIdx.x + len; i += 1024) {
		tab[i] = INF;
	}
}

__global__ void oneReduction(int * tab, int len, int mod) {

    __shared__ int begin, end;
    __shared__ int tmp_T[1024];

    if(threadIdx.x == 0) {
       begin = blockIdx.x*len;
       end = blockIdx.x*len + len;
    }

    __syncthreads();

    if(blockIdx.x % mod < mod/2) {
        for(int k = len/2; k >= 1024; k /= 2) {
            for(int g = begin; g < end; g += 2*k) {
                for(int j = g; j < g + k; j += 512) {
                    __syncthreads();

                    if(threadIdx.x < 512)
                        tmp_T[threadIdx.x] = tab[j + threadIdx.x];
                    else
                        tmp_T[threadIdx.x] = tab[j + threadIdx.x - 512 + k];

                    __syncthreads();
                    if(threadIdx.x < 512 && tmp_T[threadIdx.x] > tmp_T[threadIdx.x + 512]) {
                        tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
                        tmp_T[threadIdx.x + 512] ^= tmp_T[threadIdx.x];
                        tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
                    }

                    __syncthreads();
                    if(threadIdx.x < 512)
                        tab[j + threadIdx.x] = tmp_T[threadIdx.x];
                    else
                        tab[j + threadIdx.x - 512 + k] = tmp_T[threadIdx.x];
                }	
            }
        }

        for(int i = begin; i < begin+len; i += 1024) {
            __syncthreads();
            tmp_T[threadIdx.x] = tab[i + threadIdx.x];
            __syncthreads();
            for(int jump = 512; jump >= 1; jump /= 2) {
                if(threadIdx.x % (jump*2) < jump && threadIdx.x + jump < 1024  && tmp_T[threadIdx.x] > tmp_T[threadIdx.x + jump]) {
                    tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
                    tmp_T[threadIdx.x + jump] ^= tmp_T[threadIdx.x];
                    tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
                }
                __syncthreads();
            }
            tab[i + threadIdx.x] = tmp_T[threadIdx.x];
        }
    } else {
        for(int k = len/2; k >= 1024; k /= 2) {
            for(int g = begin; g < end; g += 2*k) {
                for(int j = g; j < g + k; j += 512) {
                    __syncthreads();
                    if(threadIdx.x < 512)
                        tmp_T[threadIdx.x] = tab[j + threadIdx.x];
                    else
                        tmp_T[threadIdx.x] = tab[j + threadIdx.x - 512 + k];

                    __syncthreads();
                    if(threadIdx.x < 512 && tmp_T[threadIdx.x] < tmp_T[threadIdx.x + 512]) {
                        tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
                        tmp_T[threadIdx.x + 512] ^= tmp_T[threadIdx.x];
                        tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
                    }

                    __syncthreads();
                    if(threadIdx.x < 512)
                        tab[j + threadIdx.x] = tmp_T[threadIdx.x];
                    else
                        tab[j + threadIdx.x - 512 + k] = tmp_T[threadIdx.x];
                }	
            }
        }

        for(int i = begin; i < begin + len; i += 1024) {
            __syncthreads();
            tmp_T[threadIdx.x] = tab[i + threadIdx.x];
            __syncthreads();
            for(int jump = 512; jump >= 1; jump /= 2) {
                if(threadIdx.x % (jump*2) < jump && threadIdx.x + jump < 1024  && tmp_T[threadIdx.x] < tmp_T[threadIdx.x + jump]) {
                    tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
                    tmp_T[threadIdx.x + jump] ^= tmp_T[threadIdx.x];
                    tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
                }
                __syncthreads();
            }
            tab[i + threadIdx.x] = tmp_T[threadIdx.x];
        }
    }


}

__global__ void oneBlock(int * tab, int len) {
    
    __shared__ int begin, end;
    __shared__ int tmp_T[1024];

    if(threadIdx.x == 0) {
       begin = blockIdx.x*len;
       end = blockIdx.x*len + len;
    }

    __syncthreads();

	//first phase

	for(int i = begin; i < end; i += 2048) {
		tmp_T[threadIdx.x] = tab[i + threadIdx.x];
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

		tab[i + threadIdx.x] = tmp_T[threadIdx.x];
		__syncthreads();

		tmp_T[threadIdx.x] = tab[i + 1024 + threadIdx.x];
		__syncthreads();

		for(int bSize = 2; bSize <= 1024; bSize *= 2) {
			for(int jump = bSize/2; jump >= 1; jump /= 2) {
				if(threadIdx.x % (jump*2) < jump && threadIdx.x + jump < 1024  && (
				  ( tmp_T[threadIdx.x] < tmp_T[threadIdx.x + jump] && threadIdx.x % (bSize*2) < bSize ) ||
				  ( tmp_T[threadIdx.x] > tmp_T[threadIdx.x + jump] && threadIdx.x % (bSize*2) >= bSize ))) {
					tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
					tmp_T[threadIdx.x + jump] ^= tmp_T[threadIdx.x];
					tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
				}
				__syncthreads();
			}
		}

		tab[i + 1024 + threadIdx.x] = tmp_T[threadIdx.x];
		__syncthreads();
	}


	// second phase

	for(int task_size = 2048; task_size < len; task_size *= 2) {
		for(int pos = begin; pos < end; pos += 2*task_size) {
			for(int k = task_size/2; k >= 1024; k /= 2) {
			    for(int lilPos = pos; lilPos < pos + task_size; lilPos += 2*k) {
                    for(int i = lilPos; i < lilPos + k; i += 512) {
                        __syncthreads();
                        if(threadIdx.x < 512)
                            tmp_T[threadIdx.x] = tab[i + threadIdx.x];
                        else
                            tmp_T[threadIdx.x] = tab[i + threadIdx.x - 512 + k];

                        __syncthreads();
                        if(threadIdx.x < 512 && tmp_T[threadIdx.x] > tmp_T[threadIdx.x + 512]) {
                            tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
                            tmp_T[threadIdx.x + 512] ^= tmp_T[threadIdx.x];
                            tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
                        }

                        __syncthreads();
                        if(threadIdx.x < 512)
                            tab[i + threadIdx.x] = tmp_T[threadIdx.x];
                        else
                            tab[i + threadIdx.x - 512 + k] = tmp_T[threadIdx.x];

                    }
                }
			}
			for(int i = pos; i < pos + task_size; i += 1024) {
				tmp_T[threadIdx.x] = tab[i + threadIdx.x];
				for(int jump = 512; jump >= 1; jump /= 2) {
				    __syncthreads();
					if(threadIdx.x % (jump*2) < jump && threadIdx.x + jump < 1024  && tmp_T[threadIdx.x] > tmp_T[threadIdx.x + jump]) {
						tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
						tmp_T[threadIdx.x + jump] ^= tmp_T[threadIdx.x];
						tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
					}
					__syncthreads();
				}
				tab[i + threadIdx.x] = tmp_T[threadIdx.x];
				__syncthreads();
			}
		}
    __syncthreads();

		for(int pos = begin + task_size; pos < end; pos += 2*task_size) {
			for(int k = task_size/2; k >= 1024; k /= 2) {
			    for(int lilPos = pos; lilPos < pos + task_size; lilPos += 2*k) {
                    for(int i = lilPos; i < lilPos + k; i += 512) {
                        __syncthreads();
                        if(threadIdx.x < 512)
                            tmp_T[threadIdx.x] = tab[i + threadIdx.x];
                        else
                            tmp_T[threadIdx.x] = tab[i + threadIdx.x - 512 + k];

                        __syncthreads();
                        if(threadIdx.x < 512 && tmp_T[threadIdx.x] < tmp_T[threadIdx.x + 512]) {
                            tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
                            tmp_T[threadIdx.x + 512] ^= tmp_T[threadIdx.x];
                            tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
                        }

                        __syncthreads();
                        if(threadIdx.x < 512)
                            tab[i + threadIdx.x] = tmp_T[threadIdx.x];
                        else
                            tab[i + threadIdx.x - 512 + k] = tmp_T[threadIdx.x];

                    }
                }
			}
			for(int i = pos; i < pos + task_size; i += 1024) {
				tmp_T[threadIdx.x] = tab[i + threadIdx.x];
				__syncthreads();
				for(int jump = 512; jump >= 1; jump /= 2) {
				    __syncthreads();
					if(threadIdx.x % (jump*2) < jump && threadIdx.x + jump < 1024  && tmp_T[threadIdx.x] < tmp_T[threadIdx.x + jump]) {
						tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
						tmp_T[threadIdx.x + jump] ^= tmp_T[threadIdx.x];
						tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + jump];
					}
					__syncthreads();
				}
				__syncthreads();
				tab[i + threadIdx.x] = tmp_T[threadIdx.x];
				__syncthreads();
			}
		}
	}
}

__global__ void oneMove(int * tab, int dist, int pow, int blocksPerTask, int period) {
	__shared__ int tmp_T[1024];
	__shared__ int begin;

	if(threadIdx.x == 0)
        begin = (blockIdx.x/blocksPerTask)*dist*2 + (blockIdx.x%blocksPerTask)*512*pow;

    __syncthreads();

    if((blockIdx.x / period) % 2 == 0) {
        for(int i = begin; i < begin + pow*512; i += 512) {
            if(threadIdx.x < 512) tmp_T[threadIdx.x] = tab[i + threadIdx.x];
            else tmp_T[threadIdx.x] = tab[i + threadIdx.x - 512 + dist];

            __syncthreads();

            if(threadIdx.x < 512 && tmp_T[threadIdx.x] > tmp_T[threadIdx.x + 512]) {
                tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
                tmp_T[threadIdx.x + 512] ^= tmp_T[threadIdx.x];
                tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
            }

            __syncthreads();

            if(threadIdx.x < 512) tab[i + threadIdx.x] = tmp_T[threadIdx.x];
            else tab[i + threadIdx.x - 512 + dist] = tmp_T[threadIdx.x];
            
            __syncthreads();
        }
    } else {
        for(int i = begin; i < begin + pow*512; i += 512) {
            if(threadIdx.x < 512) tmp_T[threadIdx.x] = tab[i + threadIdx.x];
            else tmp_T[threadIdx.x] = tab[i + threadIdx.x - 512 + dist];

            __syncthreads();

            if(threadIdx.x < 512 && tmp_T[threadIdx.x] < tmp_T[threadIdx.x + 512]) {
                tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
                tmp_T[threadIdx.x + 512] ^= tmp_T[threadIdx.x];
                tmp_T[threadIdx.x] ^= tmp_T[threadIdx.x + 512];
            }

            __syncthreads();

            if(threadIdx.x < 512) tab[i + threadIdx.x] = tmp_T[threadIdx.x];
            else tab[i + threadIdx.x - 512 + dist] = tmp_T[threadIdx.x];
            
            __syncthreads();
        }
    }
}

}
