#include"radixsort.h"
#include<cstdio>
#include<sys/mman.h>

#define LP 15485917

using namespace std;

int main(){
    int n = 1000000;
    int *c = (int*) malloc(n*sizeof(int));

    int last = 0;

    for (int j=0; j<n; ++j){
        last = (last+LP)%n;
        c[j] = last + 1;
    }
    int* d = radixsort(c, n);
    for (int j=1; j<=n; ++j){
 //       printf("%d\n", d[j-1]);
        if(d[j-1] != j) {
            printf("FAIL: %d\n", j);
            return 0;
        }
    }
    printf("OK\n");
    free(c);
    free(d);
    return 0;
}
