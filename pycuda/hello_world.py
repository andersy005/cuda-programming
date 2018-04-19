
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include <stdio.h>

    __global__ void kernel()
    {
        printf("Hello, World!\\n");
    }""")

func = mod.get_function("kernel")


func(block=(4, 1, 1))