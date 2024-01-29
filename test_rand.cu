#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <climits>
#include "timer.hpp"


const int num_blocks = 256;
const int num_threads_per_block = 256;

__device__ unsigned cuda_taus(unsigned &state) {
    int a = 3, b = 12, c = 19;
    unsigned d = 4294967295UL;
    unsigned tmp = (((state << a) ^ state) >> b);
    return state = (((state & d) << c) ^ tmp);
}
__device__ unsigned cuda_LCG(unsigned &state) {
    int a = 45, b = 3;
    return state = (a * state + b);
}

__device__ double cuda_gen_rand_num(unsigned &z_taus, unsigned &z_LCG) {
    double scaling = 1./UINT_MAX;
    return scaling * (cuda_taus(z_taus) ^ cuda_LCG(z_LCG));
}

__global__ void cuda_gen_test_seq(double *test_arr, int size, unsigned *z_taus, unsigned *z_LCG) {
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    for (int dude = blockIdx.x * blockDim.x + threadIdx.x; dude < size; dude += blockDim.x * gridDim.x){
        test_arr[dude] = cuda_gen_rand_num(z_taus[tid_global], z_LCG[tid_global]);
    }
}

int main () {
    const int arraySize = 10000000;
    double *rand_num_host = new double[arraySize];
    double *cuda_rand_num;
    unsigned *state_init = new unsigned[arraySize];
    cudaMalloc(&cuda_rand_num, sizeof(double) * arraySize);
    srand(0);

    unsigned *cuda_states_taus, *cuda_states_LCG;
    cudaMalloc(&cuda_states_taus, sizeof(unsigned)*arraySize);
    cudaMalloc(&cuda_states_LCG, sizeof(unsigned)*arraySize);

    //init states for rand generators;
    for (int i = 0; i < arraySize; ++ i) {
        state_init[i] = static_cast<unsigned int>(static_cast<double>(rand()) / RAND_MAX * UINT_MAX);
    }

    cudaMemcpy(cuda_states_taus, state_init, sizeof(unsigned)*arraySize, cudaMemcpyHostToDevice);

    for (int i = 0; i < arraySize; ++ i) {
        state_init[i] = static_cast<unsigned int>(static_cast<double>(rand()) / RAND_MAX * UINT_MAX);
    }
    cuda_gen_test_seq<<<256, 256>>>(cuda_rand_num, arraySize, cuda_states_taus, cuda_states_LCG);
    cudaMemcpy(rand_num_host, cuda_rand_num, sizeof(double) * arraySize, cudaMemcpyDeviceToHost);



    // Initialize bins
    const int numBins = 10;
    int bins[numBins] = {0};

    // Count numbers in each bin
    for (int i = 0; i < arraySize; ++i) {
        if (rand_num_host[i] >= 0 && rand_num_host[i] < 1) { 
            int binIndex = static_cast<int>(rand_num_host[i] * numBins);
            if (binIndex == numBins) binIndex = numBins - 1;
            bins[binIndex]++;
        }
    }

    printf("CHECKING STATISICAL QUALITY\n\n");
    printf("Numbers generated: %lu, in the interval betweene 0 and 1\n", arraySize);
    printf("checking number of values in the given intervals:\n\n");
    for (int i = 0; i < numBins; ++i) {

        std::cout << "Bin " << i << " (Range " << (i * 0.1) << " to " << ((i + 1) * 0.1) << "): " << bins[i] <<  " " << std::fixed << std::setprecision(2) << bins[i]*100./arraySize << "%" << std::endl;
    }

    double mean{0};
    for (int i = 0; i < arraySize; ++i) {
        mean += rand_num_host[i];
    }
    printf("\nMean: %f\n", mean/arraySize);
        
    delete[] rand_num_host;
    delete[] state_init;
    cudaFree(cuda_rand_num);
    cudaFree(cuda_states_LCG);
    cudaFree(cuda_states_taus);

    return EXIT_SUCCESS;
}