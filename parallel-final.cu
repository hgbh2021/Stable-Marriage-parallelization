#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <cuda.h>

void initialize(int *menmatching, int *womenmatching, int *menpref, int *womanlock, int n) {
    int i;
    for(i=0; i<=n; i++) {
        menmatching[i] = -1;
        menpref[i] = 1;
        womenmatching[i] = -1;
        womanlock[i] = 0;
    }
}

__global__ void stable_matching(int n, int *d_men, int *d_women,
        int *d_menmatching, int *d_womenmatching, int *d_menpref, int *d_matched, int *d_matched_, int *d_womanlock) {
    int j = threadIdx.x + 1, idx, ct=0;
    while(1) {
        __syncthreads();
        if(*d_matched_ == 0) break;
        if(*d_matched_ == 1 && j <= n && d_menmatching[j] == -1) {
            idx = d_men[j*(n+1) + d_menpref[j]];
            *d_matched = 0;
            // locking mechanism
            bool isSet = false;
            do {
                if(isSet = atomicCAS(&d_womanlock[idx], 0, 1) == 0) {
                    if(d_womenmatching[idx] == -1) {
                        d_womenmatching[idx] = j;
                        d_menmatching[j] = idx;
                    }
                    else if(d_women[idx*(n+1) + d_womenmatching[idx]] > d_women[idx*(n+1) + j]) {
                        d_menmatching[j] = idx;
                        d_menmatching[d_womenmatching[idx]] = -1;
                        d_womenmatching[idx] = j;
                    }
                }
                if(isSet) {
                    atomicCAS(&d_womanlock[idx], 1, 0);
                }
            } while(!isSet);
            d_menpref[j]++;
        }
        __syncthreads();
        if(j == 1 && *d_matched == 1) {
            *d_matched_ = 0;
        }
        else if(j == 1 && *d_matched == 0) {
            *d_matched = 1;
        }
        ct++;
    }
    __syncthreads();
}

int main()
{
    int n,i,j,k;
    int *d_matched, *d_matched_;
    int *men, *women;
    int *menmatching, *womenmatching, *menpref, *womanlock;
    int *d_men, *d_women;
    int *d_menmatching, *d_womenmatching, *d_menpref, *d_womanlock;
    clock_t beg, end;
    double read_time;

    scanf("%d",&n);
    men = (int *) malloc((n+1)*(n+1)*sizeof(int));
    menmatching = (int *) malloc((n+1)*sizeof(int));
    menpref = (int *) malloc((n+1)*sizeof(int));
    women = (int *) malloc((n+1)*(n+1)*sizeof(int));
    womenmatching = (int *) malloc((n+1)*sizeof(int));
    womanlock = (int *) malloc((n+1)*sizeof(int));

    cudaMalloc(&d_men, (n+1)*(n+1)*sizeof(int));
    cudaMalloc(&d_menmatching, (n+1)*sizeof(int));
    cudaMalloc(&d_menpref, (n+1)*sizeof(int));
    cudaMalloc(&d_women, (n+1)*(n+1)*sizeof(int));
    cudaMalloc(&d_womenmatching, (n+1)*sizeof(int));
    cudaMalloc(&d_womanlock, (n+1)*sizeof(int));
    cudaMalloc(&d_matched, sizeof(int));
    cudaMalloc(&d_matched_, sizeof(int));

    initialize(menmatching, womenmatching, menpref, womanlock, n);

    beg = clock();
    for(i=1; i<=n; i++) {
        for(j=0; j<=n; j++) {
            scanf("%d", &men[i*(n+1) + j]);
        }
    }

    for(i=1; i<=n; i++) {
        for(j=0; j<=n; j++) {
            scanf("%d", &k);
            women[i*(n+1) + k] = j;
        }
    }
    end = clock();
    read_time = ((double)(end-beg) * 1000000)/CLOCKS_PER_SEC;
    printf("time for reading : %f us, ", read_time);

    cudaMemcpy(d_men, men, (n+1)*(n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_menpref, menpref, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_menmatching, menmatching, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_women, women, (n+1)*(n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_womanlock, womanlock, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_womenmatching, womenmatching, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    int matched = 1;
    cudaMemcpy(d_matched, &matched, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matched_, &matched, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float compute_time = 0;
    cudaEventRecord(start,0);

    stable_matching <<< 1, n >>>(n, d_men, d_women, d_menmatching, d_womenmatching, d_menpref, d_matched, d_matched_, d_womanlock);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&compute_time, start, stop);

    cudaMemcpy(menmatching, d_menmatching, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    printf("time for computation : %f us\n", compute_time*1000);

    for(j=1;j<=n;j++)
        printf("%d %d\n", j, menmatching[j]);

    free(men); 
    free(menpref); 
    free(menmatching); 
    free(women);
    free(womenmatching); 
    free(womanlock);
    cudaFree(&d_men); 
    cudaFree(&d_women); 
    cudaFree(&d_matched); 
    cudaFree(&d_matched_);
    cudaFree(&d_menmatching); 
    cudaFree(&d_menpref); 
    cudaFree(&d_womenmatching); 
    cudaFree(&d_womanlock);

    return 0;
}