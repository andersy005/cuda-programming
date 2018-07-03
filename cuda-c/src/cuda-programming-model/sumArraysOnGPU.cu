
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}


__global__ void sumArraysOnDevice(float *A, float *B, float *C, const int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];

}


void initialData(float *ip, int size){
    // generate different seed for random number 
    time_t t;
    srand((unsigned int) time (&t));
    
    for (int i=0; i<size; i++){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}


void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for (int idx=0; idx<N; idx++){
        C[idx] = A[idx] + B[idx];
    }
}



void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    int match = 1;
    for (int i = 0; i < N; i++){
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n",
                   hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match. \n\n");
}


int main(int argc, char **argv){
    
    printf("%s Starting...\n", argv[0]);
    
    // malloc host memory
    int nElem = 1 <<24;
    size_t nBytes = nElem * sizeof(float);
    
    
    // initialize data at host side
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    
    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    
    // malloc device global memory 
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    
    // Use cudaMemcpy to transfer the data from the host memory to the GPU global memory with the
    // parameter cudaMemcpyHostToDevice specifying the transfer direction.
    
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    
    // invoke kernel at host side
    int iLen = 128;
    dim3 block(iLen);
    dim3 grid((nElem+block.x-1)/block.x);
    
    double iStart = cpuSecond();
    sumArraysOnDevice<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    double iElaps = cpuSecond() - iStart;
    printf("sumArraysOnGPU <<<%d,%d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);
    //printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);
    
    // copy kernel result back to host side 
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    
    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    
    for (int i=0; i<10; i++){
         printf("%f + %f = %f \n", h_A[i], h_B[i], hostRef[i]);

    }
    
    // check device results
    checkResult(hostRef, gpuRef, nElem);
    
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    
    // use cudaFree to release the memory used on the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
    
    return (0);
}