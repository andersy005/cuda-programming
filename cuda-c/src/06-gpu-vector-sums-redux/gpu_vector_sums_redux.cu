#include <stdio.h>
#include "../utils/common.h"

#define N 100

__global__ void add(int *a, int *b, int *c){
    int tid = threadIdx.x;  // handle the data at this index
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}


int main(void){
    int a[N], b[N], c[N];
    int *device_a, *device_b, *device_c;

    // Allocate the memory on the GPU
    HANDLE_ERROR(cudaMalloc((void**)&device_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&device_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&device_c, N * sizeof(int)));


    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++){
        a[i] = -i;
        b[i] = i * i;
    }


    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(device_a, a, N * sizeof(int), 
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(device_b, b, N * sizeof(int),
                            cudaMemcpyHostToDevice));

    // Execute the kernel with N parallel blocks
    add<<<1, N>>>(device_a, device_b, device_c);

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR(cudaMemcpy(c, device_c, N * sizeof(int),
                           cudaMemcpyDeviceToHost));

    // display the results
    for(int i=0; i<N; i++){
        printf("%d + %d = % d\n", a[i], b[i], c[i]);
    }

    // free the memory allocated on the GPU
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 0;
}