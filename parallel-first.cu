#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <cuda.h>

void initialize(int *menmatch, int *womenmatch, int *menpre, int *womanlock, int n) {
    int i;
    for(i=0; i<=n; i++) {
        menmatch[i] = -1;
        womenmatch[i] = -1;
        menpre[i] = 1;
        womanlock[i] = 0;
    }
}

// kernel-1 implementation
__global__ void stable_matching(int n, int *d_men, int *d_women,
        int *d_menmatch, int *d_womenmatch, int *d_menpre, int *d_matched, int *d_womanlock) {
    int j = threadIdx.x + 1, indx;
    indx = d_men[j*(n+1) + d_menpre[j]];
    if(j <= n && d_menmatch[j] == -1) {
        bool isSet = false;
        do {
            if(isSet = atomicCAS(&d_womanlock[indx], 0, 1) == 0) {
                if(d_womenmatch[indx] == -1) {
                    d_menmatch[j] = indx;
                    d_womenmatch[indx] = j;
                    atomicAdd(d_matched, 1);
                }
                else if(d_women[indx*(n+1) + d_womenmatch[indx]] > d_women[indx*(n+1) + j]) {
                    d_menmatch[j] = indx;
                    d_menmatch[d_womenmatch[indx]] = -1;
                    d_womenmatch[indx] = j;
                }
            }
            if(isSet) {
                atomicCAS(&d_womanlock[indx], 1, 0);
            }
        } while(!isSet);
        d_menpre[j]++;
    }
}

int main()
{
    int n,i,j,k;
    int matched=0;
    int *d_matched;
    int *men, *women;
    int *menmatch, *womenmatch, *menpre, *womanlock;
    int *d_men, *d_women;
    int *d_menmatch, *d_womenmatch, *d_menpre, *d_womanlock;
    clock_t beg, end;
    double read_time;

    scanf("%d",&n);
    men = (int *) malloc((n+1)*(n+1)*sizeof(int));
    menmatch = (int *) malloc((n+1)*sizeof(int));
    menpre = (int *) malloc((n+1)*sizeof(int));
    women = (int *) malloc((n+1)*(n+1)*sizeof(int));
    womenmatch = (int *) malloc((n+1)*sizeof(int));
    womanlock = (int *) malloc((n+1)*sizeof(int));

    cudaMalloc(&d_men, (n+1)*(n+1)*sizeof(int));
    cudaMalloc(&d_menmatch, (n+1)*sizeof(int));
    cudaMalloc(&d_menpre, (n+1)*sizeof(int));
    cudaMalloc(&d_women, (n+1)*(n+1)*sizeof(int));
    cudaMalloc(&d_womenmatch, (n+1)*sizeof(int));
    cudaMalloc(&d_womanlock, (n+1)*sizeof(int));
    cudaMalloc(&d_matched, sizeof(int));

    initialize(menmatch, womenmatch, menpre, womanlock, n);

    beg = clock();
    for(i=1; i<=n; i++) {
        for(j=0; j<=n; j++) scanf("%d", &men[i*(n+1) + j]);
    }

    for(i=1; i<=n; i++) {
        for(j=0; j<=n; j++) {
            scanf("%d", &k);
            women[i*(n+1) + k] = j;
        }
    }
    end = clock();
    read_time = ((double)(end-beg) * 1000000)/CLOCKS_PER_SEC;
    printf("read time : %f us, ", read_time);

    cudaMemcpy(d_men, men, (n+1)*(n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_menmatch, menmatch, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_menpre, menpre, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_women, women, (n+1)*(n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_womenmatch, womenmatch, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_womanlock, womanlock, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matched, &matched, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float compute_time = 0;
    cudaEventRecord(start,0);

    int ct=0;
    while(matched != n) {
        ct++;
        stable_matching <<< 1, n >>>(n, d_men, d_women, d_menmatch, d_womenmatch, d_menpre, d_matched, d_womanlock);
        cudaMemcpy(&matched, d_matched, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&compute_time, start, stop);

    cudaMemcpy(menmatch, d_menmatch, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    printf("time for computation : %f us\n", compute_time*1000);
    

    for(j=1;j<=n;j++)
        printf("%d %d\n", j, menmatch[j]);

    free(men); 
    free(menmatch); 
    free(menpre); 
    free(women);
    free(womenmatch); 
    free(womanlock);
    cudaFree(&d_men); 
    cudaFree(&d_menmatch); 
    cudaFree(&d_menpre); 
    cudaFree(&d_women); 
    cudaFree(&d_womenmatch); 
    cudaFree(&d_womanlock);
    cudaFree(&d_matched);

    return 0;
}