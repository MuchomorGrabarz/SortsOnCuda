#include"bitonic.h"

#include<cstdlib>
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
    int* d = bitonic(c, n);
    for (int j=1; j<=n; ++j){
  //      printf("%d\n", d[j-1]);
        if(d[j-1] != j) {
            printf("FAIL: %d : %d\n", j, d[j]);
            for(int k = j + 1; k<=j+10; k++) printf("tab[%d] = %d\n", k, d[k]);
            return 0;
        }
    }
    printf("OK\n");
    free(c);
    free(d);
    return 0;
}
