int main(void){
    CPUBitmap bitmap (DIM, DIM);
    unsigned char *device_bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&device_bitmap, bitmap.image_size()));

    dim3 grid(DIM, DIM);

    kernel<<<grid, 1>>>(device_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), device_bitmap,
                            bitmap.image_size(), cudaMemcpyDeviceToHost));

    bitmap.display_and_exit();
    cudaFree(device_bitmap);

    return 0; 
}