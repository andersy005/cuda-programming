#include <stdio.h>

/* Alerts the compiler that a function should be compiled to run on a device
instead of the host.
*/
__global__ void kernel(void) {

}

int main(void){

	kernel<<<1,1>>>();
	/* Angle brackets denote arguments we plan to pass to the runtime system. 
	These are not arguments to the device code. */
	printf("Hello, World!\n");
	return 0;

}


