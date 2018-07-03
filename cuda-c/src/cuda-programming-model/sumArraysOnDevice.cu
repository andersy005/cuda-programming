
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

__global__ void sumArraysOnDevice(float *A, float *B, float *C){
    int idx = threadIdx.x;
    C[idx] = A[idx] + B[idx];

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



void checkResult(float *h_C, float *result, const int N){
    double epsilon = 1.0E-8;
    int match = 1;
    for (int i = 0; i < N; i++){
        if (abs(h_C[i] - result[i]) > epsilon){
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n",
                   h_C[i], result[i], i);
            break;
        }
    }
    if (match) printf("Arrays match. \n\n");
}


int main(int argc, char **argv){
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);
    
    float *h_A, *h_B, *h_C, *result;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    result = (float *)malloc(nBytes);
    
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    
    // Use cudaMemcpy to transfer the data from the host memory to the GPU global memory with the
    // parameter cudaMemcpyHostToDevice specifying the transfer direction.
    
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    
    
    
    sumArraysOnDevice<<<1, nElem>>>(d_A, d_B, d_C);
    sumArraysOnHost(h_A, h_B, result, nElem);
    
    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
    
    for (int i=0; i<10; i++){
         printf("%f + %f = %f \n", h_A[i], h_B[i], h_C[i]);

    }
    
    checkResult(h_C, result, nElem);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(result);
    
    // use cudaFree to release the memory used on the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
    
    return (0);
}