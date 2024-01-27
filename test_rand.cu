#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <climits>
#include "timer.hpp"

#define MOD 4294967296 // 2^32

const int num_blocks = 256;
const int num_threads_per_block = 256;

// __device__ unsigned long cuda_LCG1(unsigned long &state) {
//     const unsigned long a = 163936, c = 34048436;
//     state = (a * state + c) % MOD;
//     return state;
// }

// __device__ unsigned long cuda_LCG2(unsigned long &state) {
//     const unsigned long a = 127436, c = 33292876;
//     state = (a * state + c) % MOD;
//     return state;
// }
// __device__ unsigned long cuda_LCG3(unsigned long &state) {
//     const unsigned long a = 1633936, c = 92387456;
//     state = (a * state + c) % MOD;
//     return state;
// }
// __global__ void cuda_mult_LCG(double *rand_num, int size) {
//     int tid_global = threadIdx.x + blockIdx.x * blockDim.x;
//     unsigned long state1 = tid_global, state2 = tid_global+10, state3 = tid_global+2;
//     double res_double{0};
//     for (int i = 0; i < 10; ++i) {
//         unsigned long r1 = cuda_LCG1(state1);
//         unsigned long r2 = cuda_LCG2(state2);
//         unsigned long r3 = cuda_LCG3(state3);

//         unsigned long result = (r1 +r2 +r3) % MOD;
//         state1 = result + 1;
//         state2 = result + 20;
//         state3 = result + 3;
//         res_double = static_cast<double>(result)/MOD;
//     }
//     rand_num[tid_global] = res_double;    
// }

__device__ float generate_combined_random_number(unsigned int &lcg_state, unsigned int &taus_state) {
    // LCG parameters (example values)
    const unsigned int a = 1664525, c = 1013904223, m = 4294967296;

    // Tausworthe parameters (example values)
    const int S1 = 13, S2 = 19, S3 = 12;
    const unsigned int M = 4294967294U;

    // Generate random number using LCG
    lcg_state = (a * lcg_state + c) % m;

    // Generate random number using Tausworthe
    unsigned b = (((taus_state << S1) ^ taus_state) >> S2);
    taus_state = (((taus_state & M) << S3) ^ b);

    // Combine and scale the result
    unsigned int combined_random = lcg_state ^ taus_state;
    return static_cast<double>(combined_random) / static_cast<double>(UINT_MAX);
}

__global__ void generate_random_numbers(float *random_numbers, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize states for LCG and Taus
    unsigned int lcg_state = seed + idx;
    unsigned int taus_state = seed ^ idx;

    // Call device function to get a single random number
    random_numbers[idx] = generate_combined_random_number(lcg_state, taus_state);
}

int main () {
    const int arraySize = 256 * 256;
    float *rand_num_host = new float[arraySize];
    float *cuda_rand_num;
    cudaMalloc(&cuda_rand_num, sizeof(float) * arraySize);

    generate_random_numbers<<<256, 256>>>(cuda_rand_num, 2873456);
    cudaMemcpy(rand_num_host, cuda_rand_num, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);

    // Initialize bins
    const int numBins = 10;
    int bins[numBins] = {0};

    // Count numbers in each bin
    for (int i = 0; i < arraySize; ++i) {
        if (rand_num_host[i] >= 0 && rand_num_host[i] < 1) { // Ensure the number is within [0, 1)
            int binIndex = static_cast<int>(rand_num_host[i] * numBins);
            if (binIndex == numBins) binIndex = numBins - 1; // Edge case for 1.0
            bins[binIndex]++;
        }
    }

    // Print the count in each bin
    for (int i = 0; i < numBins; ++i) {
        std::cout << "Bin " << i << " (Range " << (i * 0.1) << " to " << ((i + 1) * 0.1) << "): " << bins[i] << std::endl;
    }
        
    delete[] rand_num_host;
    cudaFree(cuda_rand_num);

    return EXIT_SUCCESS;
}