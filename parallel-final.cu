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

void initialize(int *menmatching, int *womenmatching, int *menpref, int *womanlock, int n) {
    int i;
    for(i = 0; i <= n; i++) {
        menmatching[i] = -1;
        menpref[i] = 1;
        womenmatching[i] = -1;
        womanlock[i] = 0;
    }
}

// Kernel-2 (persistent kernel): Entire matching loop runs inside a single kernel.
// All threads stay alive and participate in every __syncthreads() barrier.
// Uses shared memory flag to coordinate termination.
__global__ void stable_matching(int n, int *d_men, int *d_women,
        int *d_menmatching, int *d_womenmatching, int *d_menpref, int *d_womanlock) {

    int j = threadIdx.x + 1;

    // Shared flag: 1 = keep going, 0 = all matched, terminate
    __shared__ int s_any_free;

    // Initialize shared flag
    if(j == 1) {
        s_any_free = 1;
    }
    __syncthreads();

    while(s_any_free) {
        // --- Proposal phase ---
        // Only unmatched men within valid range propose
        if(j <= n && d_menmatching[j] == -1) {
            int idx = d_men[j * (n + 1) + d_menpref[j]];

            // Acquire spinlock on the target woman
            bool isSet = false;
            do {
                isSet = (atomicCAS(&d_womanlock[idx], 0, 1) == 0);
                if(isSet) {
                    if(d_womenmatching[idx] == -1) {
                        // Woman is free — accept proposal
                        d_womenmatching[idx] = j;
                        d_menmatching[j] = idx;
                    }
                    else if(d_women[idx * (n + 1) + d_womenmatching[idx]] > d_women[idx * (n + 1) + j]) {
                        // Woman prefers new proposer — dump current partner
                        d_menmatching[d_womenmatching[idx]] = -1;  // dump old
                        d_menmatching[j] = idx;
                        d_womenmatching[idx] = j;
                    }
                    // Release lock
                    atomicExch(&d_womanlock[idx], 0);
                }
            } while(!isSet);

            // Advance preference pointer
            d_menpref[j]++;
        }

        // All threads must reach this barrier (no divergent break above)
        __syncthreads();

        // --- Termination check phase ---
        // Thread 1 optimistically sets "all matched"
        if(j == 1) {
            s_any_free = 0;
        }
        __syncthreads();

        // Any unmatched man sets the flag back to 1
        if(j <= n && d_menmatching[j] == -1) {
            s_any_free = 1;  // shared mem — visible to whole block after next sync
        }
        __syncthreads();

        // Now s_any_free == 0 only if nobody is unmatched → loop exits
    }
}

int main()
{
    int n, i, j, k;
    int *men, *women;
    int *menmatching, *womenmatching, *menpref, *womanlock;
    int *d_men, *d_women;
    int *d_menmatching, *d_womenmatching, *d_menpref, *d_womanlock;
    clock_t beg, end;
    double read_time;

    scanf("%d", &n);

    if(n > 1024) {
        fprintf(stderr, "Error: n=%d exceeds max thread block size (1024).\n", n);
        return 1;
    }

    men = (int *) malloc((n + 1) * (n + 1) * sizeof(int));
    menmatching = (int *) malloc((n + 1) * sizeof(int));
    menpref = (int *) malloc((n + 1) * sizeof(int));
    women = (int *) malloc((n + 1) * (n + 1) * sizeof(int));
    womenmatching = (int *) malloc((n + 1) * sizeof(int));
    womanlock = (int *) malloc((n + 1) * sizeof(int));

    CUDA_CHECK(cudaMalloc(&d_men, (n + 1) * (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_menmatching, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_menpref, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_women, (n + 1) * (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_womenmatching, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_womanlock, (n + 1) * sizeof(int)));

    initialize(menmatching, womenmatching, menpref, womanlock, n);

    beg = clock();
    for(i = 1; i <= n; i++) {
        for(j = 0; j <= n; j++) {
            scanf("%d", &men[i * (n + 1) + j]);
        }
    }

    for(i = 1; i <= n; i++) {
        for(j = 0; j <= n; j++) {
            scanf("%d", &k);
            women[i * (n + 1) + k] = j;
        }
    }
    end = clock();
    read_time = ((double)(end - beg) * 1000000) / CLOCKS_PER_SEC;
    printf("time for reading : %f us, ", read_time);

    CUDA_CHECK(cudaMemcpy(d_men, men, (n + 1) * (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_menpref, menpref, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_menmatching, menmatching, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_women, women, (n + 1) * (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_womanlock, womanlock, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_womenmatching, womenmatching, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float compute_time = 0;
    CUDA_CHECK(cudaEventRecord(start, 0));

    stable_matching<<<1, n>>>(n, d_men, d_women, d_menmatching, d_womenmatching, d_menpref, d_womanlock);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&compute_time, start, stop));

    CUDA_CHECK(cudaMemcpy(menmatching, d_menmatching, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    printf("time for computation : %f us\n", compute_time * 1000);

    for(j = 1; j <= n; j++)
        printf("%d %d\n", j, menmatching[j]);

    free(men);
    free(menpref);
    free(menmatching);
    free(women);
    free(womenmatching);
    free(womanlock);
    cudaFree(d_men);
    cudaFree(d_women);
    cudaFree(d_menmatching);
    cudaFree(d_menpref);
    cudaFree(d_womenmatching);
    cudaFree(d_womanlock);

    return 0;
}