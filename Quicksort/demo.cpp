#include"quicksort.h"
#include<cstdio>
#include<sys/mman.h>

#define LP 15485917

using namespace std;

int main(){
    int n = 100000000;
    int *c = (int*) malloc(n*sizeof(int));
    
    int last = 0;

    for (int j=0; j<n; ++j){
        last = (last+LP)%n;
        c[j] = last + 1;
    }
    int* d = quicksort(c, n);
    for (int j=1; j<=n; ++j){
        if(d[j-1] != j) {
            for(int i = 0; i<=99; i++)printf("FAIL: %d : %d\n", j+i, d[j+i-1]);
            return 0;
        }
    }
    printf("OK\n");
    free(c);
    free(d);
    return 0;
}
