#include <stdio.h>
#include "../utils/common.h"
#include "../utils/cpu_bitmap.h"

#define DIM 1000


/* cuComplex structure that defines a method for storing a complex number
 with single precision floating-point components. The structure also defines
 addition and multiplication operators as well as a function to return 
 the magnitude of the complex value. 
*/

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};


// Code that determines whether a point is in or out of the 
// Julia Set. 

__device__ int julia(int x, int y){
    const float scale= 1.5;
    float jx = scale * (float)(DIM/2 - x) / (DIM/2);
    float jy = scale * (float)(DIM/2 - y) / (DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++){
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(unsigned char *ptr){
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    // compute linear offset with help of built-iin variable, gridDim
    // This variable is a constant across all blocks and simply holds the 
    // dimensions of the grid that was launched. 
    // In this example, it will always be the value (DIM, DIM)
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    int juliaValue = julia(x, y);
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;

}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};


int main(void){

    // Create DIM x DIM bitmap image using utility library
    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );


    // Because the computation will be done on a GPU, declare a pointer to hold a copy
    // of data on the device 
    unsigned char *device_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&device_bitmap, bitmap.image_size()));
 
    // type dim3 is a CUDA runtime type that represents a 3-D (with z=1)  
    // tuple that will be used to specify the size of our launch 
    dim3 grid(DIM, DIM);

    kernel<<<grid, 1>>>(device_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), device_bitmap,
                            bitmap.image_size(), cudaMemcpyDeviceToHost));

    
    cudaFree(device_bitmap);
    bitmap.display_and_exit();

}
