#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <cuda.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

void initialize(int *menmatch, int *womenmatch, int *menpre, int *womanlock, int n) {
    int i;
    for(i = 0; i <= n; i++) {
        menmatch[i] = -1;
        womenmatch[i] = -1;
        menpre[i] = 1;
        womanlock[i] = 0;
    }
}

// Kernel-1: One kernel launch per proposal round.
// Each unmatched man proposes to his next preferred woman.
// Uses atomicCAS-based spinlock to protect per-woman critical sections.
__global__ void stable_matching(int n, int *d_men, int *d_women,
        int *d_menmatch, int *d_womenmatch, int *d_menpre, int *d_any_free, int *d_womanlock) {
    int j = threadIdx.x + 1;

    // Only unmatched men propose
    if(j <= n && d_menmatch[j] == -1) {
        int indx = d_men[j * (n + 1) + d_menpre[j]];

        // Acquire spinlock on the target woman
        bool isSet = false;
        do {
            isSet = (atomicCAS(&d_womanlock[indx], 0, 1) == 0);
            if(isSet) {
                if(d_womenmatch[indx] == -1) {
                    // Woman is free — accept proposal
                    d_menmatch[j] = indx;
                    d_womenmatch[indx] = j;
                }
                else if(d_women[indx * (n + 1) + d_womenmatch[indx]] > d_women[indx * (n + 1) + j]) {
                    // Woman prefers new proposer — dump current partner
                    d_menmatch[d_womenmatch[indx]] = -1;  // dump old partner
                    d_menmatch[j] = indx;
                    d_womenmatch[indx] = j;
                }
                // Release lock
                atomicExch(&d_womanlock[indx], 0);
            }
        } while(!isSet);

        // Advance preference pointer (man should never re-propose to same woman)
        d_menpre[j]++;
    }

    // Wait for all threads to finish proposals
    __syncthreads();

    // After all proposals are done, check if this man is still unmatched
    // (could have been displaced by another man during this round)
    if(j <= n && d_menmatch[j] == -1) {
        *d_any_free = 1;
    }
}

int main()
{
    int n, i, j, k;
    int *d_any_free;
    int *men, *women;
    int *menmatch, *womenmatch, *menpre, *womanlock;
    int *d_men, *d_women;
    int *d_menmatch, *d_womenmatch, *d_menpre, *d_womanlock;
    clock_t beg, end;
    double read_time;

    scanf("%d", &n);

    if(n > 1024) {
        fprintf(stderr, "Error: n=%d exceeds max thread block size (1024).\n", n);
        return 1;
    }

    men = (int *) malloc((n + 1) * (n + 1) * sizeof(int));
    menmatch = (int *) malloc((n + 1) * sizeof(int));
    menpre = (int *) malloc((n + 1) * sizeof(int));
    women = (int *) malloc((n + 1) * (n + 1) * sizeof(int));
    womenmatch = (int *) malloc((n + 1) * sizeof(int));
    womanlock = (int *) malloc((n + 1) * sizeof(int));

    CUDA_CHECK(cudaMalloc(&d_men, (n + 1) * (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_menmatch, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_menpre, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_women, (n + 1) * (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_womenmatch, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_womanlock, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_any_free, sizeof(int)));

    initialize(menmatch, womenmatch, menpre, womanlock, n);

    beg = clock();
    for(i = 1; i <= n; i++) {
        for(j = 0; j <= n; j++) scanf("%d", &men[i * (n + 1) + j]);
    }

    for(i = 1; i <= n; i++) {
        for(j = 0; j <= n; j++) {
            scanf("%d", &k);
            women[i * (n + 1) + k] = j;
        }
    }
    end = clock();
    read_time = ((double)(end - beg) * 1000000) / CLOCKS_PER_SEC;
    printf("read time : %f us, ", read_time);

    CUDA_CHECK(cudaMemcpy(d_men, men, (n + 1) * (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_menmatch, menmatch, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_menpre, menpre, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_women, women, (n + 1) * (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_womenmatch, womenmatch, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_womanlock, womanlock, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float compute_time = 0;
    CUDA_CHECK(cudaEventRecord(start, 0));

    int any_free = 1;
    int ct = 0;
    while(any_free) {
        any_free = 0;
        CUDA_CHECK(cudaMemcpy(d_any_free, &any_free, sizeof(int), cudaMemcpyHostToDevice));

        stable_matching<<<1, n>>>(n, d_men, d_women, d_menmatch, d_womenmatch, d_menpre, d_any_free, d_womanlock);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&any_free, d_any_free, sizeof(int), cudaMemcpyDeviceToHost));
        ct++;
    }

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&compute_time, start, stop));

    CUDA_CHECK(cudaMemcpy(menmatch, d_menmatch, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    printf("time for computation : %f us (rounds: %d)\n", compute_time * 1000, ct);

    for(j = 1; j <= n; j++)
        printf("%d %d\n", j, menmatch[j]);

    free(men);
    free(menmatch);
    free(menpre);
    free(women);
    free(womenmatch);
    free(womanlock);
    cudaFree(d_men);
    cudaFree(d_menmatch);
    cudaFree(d_menpre);
    cudaFree(d_women);
    cudaFree(d_womenmatch);
    cudaFree(d_womanlock);
    cudaFree(d_any_free);

    return 0;
}