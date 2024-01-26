#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "timer.hpp"

#define MOD 4294967296;

const int num_blocks = 256;
const int num_threads_per_block = 256;

__device__ unsigned long cuda_LCG1(unsigned long &state) {
    const unsigned long a = 163936, c = 340658796;
    state = (a * state + c) % MOD;
    return state;
}

__device__ unsigned long cuda_LCG2(unsigned long &state) {
    const unsigned long a = 127436, c = 3495876;
    state = (a * state + c) % MOD;
    return state;
}
__device__ unsigned long cuda_LCG3(unsigned long &state) {
    const unsigned long a = 163936, c = 92387456;
    state = (a * state + c) % MOD;
    return state;
}
__global__ void cuda_mult_LCG(double *rand_num, int size) {
    int tid_global = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned long state1 = tid_global, state2 = tid_global+1, state3 = tid_global+2;
    double res_double{0};
    for (int i = 0; i < 10; ++i) {
        unsigned long r1 = cuda_LCG1(state1);
        unsigned long r2 = cuda_LCG2(state2);
        unsigned long r3 = cuda_LCG3(state3);

        unsigned long result = (r1 +r2 +r3) % MOD;
        state1 = result + 1;
        state2 = result + 2;
        state3 = result + 3;
        res_double = static_cast<double>(result)/MOD;
    }
    rand_num[tid_global] = res_double;    
}

int main () {
    double *rand_num_host = new double[256*256];
    double *cuda_rand_num;
    cudaMalloc(&cuda_rand_num, sizeof(double)*256*256)
    
    cuda_mult_LCG<<<256, 256>>>(cuda_rand_num, 256*256);
    cudaMemcpy(rand_num_host, cuda_rand_num, sizeof(double)*256*256, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 256*256; ++i)
        printf("%f\n", rand_num_host[i]);


    return EXIT_SUCCESS;
}