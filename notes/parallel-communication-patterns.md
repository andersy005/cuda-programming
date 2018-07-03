# Parallel Communication Patterns

- Parallel computing is all about many threads solving a problem by working together.
- This is all about communication.
- In CUDA, this communication takes place in memory.

## Map and Gather

- With map, you've got many data elements (e.g., elements of an array, entries in a matrix, or pixels in an image).
- And you are going to do the same function, or computational task, on each piece of data. 
- There is a 1 to 1 correspondence between input and output. So, map is very efficient on GPUs