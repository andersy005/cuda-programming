#include <stdio.h>
#include "../utils/common.h"

__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main(){
    int c;
    int *device_c;

    HANDLE_ERROR(cudaMalloc((void**)&device_c, sizeof(int)));

    add<<<1,1>>>(2, 7, device_c);

    HANDLE_ERROR(cudaMemcpy(&c,
                           device_c,
                           sizeof(int),
                           cudaMemcpyDeviceToHost));

    printf(" 2 + 7 = %d\n", c);
    cudaFree(device_c);

    return 0;
}