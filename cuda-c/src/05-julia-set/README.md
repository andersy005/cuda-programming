# Julia Set 

## To compile the application:


In some cases, we need to add ` -lglut -lGLU -lGL` on the link line

`$ nvcc -o julia_set julia_set.cu -lglut -lGLU -lGL`

## To profile and run the code, do the following:

`$ nvprof --unified-memory-profiling off ./julia_set`

![](https://i.imgur.com/kocLDtn.gif)