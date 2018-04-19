# Passing Parameters

To compile the program, run the following:

`$ nvcc -o 02-passing-params 02-passing-params.cu`

To run the compile program:

`$ ./02-passing-params`

To run the profiler:

`$ nvprof --unified-memory-profiling off ./passing_params`

[![asciicast](https://asciinema.org/a/mIFzam2aaqraUV6NxtWH7zpPc.png)](https://asciinema.org/a/mIFzam2aaqraUV6NxtWH7zpPc)

# Summary

- We can pass parameters to a kernel as we would with any C function.
- We need to allocate memory to do anything useful on a device, such as return values to the host.

Restrictions on the usage of device pointers are as follows:

- We **can** pass pointers allocated with `cudaMalloc()` to functions that execute on the device.
- We **can** use pointers allocated with `cudaMalloc()` to read or write memory from code that executes on the device.
- We **can** pass pointers allocated with `cudaMalloc()` to functions that execute on the host.
- We **cannot** use pointers allocated with `cudaMalloc()` to read or write memory from code that executes on the host. 
