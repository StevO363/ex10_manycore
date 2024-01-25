#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "timer.hpp"

const int num_blocks = 256;
const int num_threads_per_block = 256;

//
// Data container for simulation input
//
typedef struct
{

    size_t population_size;    // Number of people to simulate

    //// Configuration
    int pushback_threshold;    // Number of fake news believers required for mainstream pushback
    int starting_fakenews;     // Number of people believing in fake news at the start of the year
    double recovery_rate;      // Rate of a person giving up on fake news and becoming reasonable again

    // for each day:
    int    *contacts_per_day;           // number of other persons met each day to whom the fake news may be passed on
    double *transmission_probability;   // how likely it is to pass on the fake news to another person

} SimInput_t;


//
// Constructor for the simulation input structure
//
void init_input(SimInput_t *input)
{
    // input->population_size = 8916845;  // Austria's population in 2020 according to Statistik Austria
    input->population_size = 20000;  // Austria's population in 2020 according to Statistik Austria

    input->pushback_threshold   = 20000;   // as soon as we have 20k fake news believers, the general public starts to push back
    input->starting_fakenews    = 100;
    input->recovery_rate        = 0.01;

    input->contacts_per_day = (int*)malloc(sizeof(int) * 365);
    input->transmission_probability = (double*)malloc(sizeof(double) * 365);
    for (int day = 0; day < 365; ++day) {
    input->contacts_per_day[day] = 2;             // arbitrary assumption of six possible transmission contacts per person per day, all year
    input->transmission_probability[day] = 0.1;   // 10 percent chance of convincing a contact of fake news
    }
}



//
// Data container for simulation output
//
typedef struct
{
// for each day:
int *active_fakenews_believers;     // number of people currently believing in fake news

// for each person:
double *fakenews_belief_strength;    // 0 if completely fake-news absent, 1 if fully believing in fake news. A person is considered to be fake news convinced if this number is larger than 0.5 (i.e. more belief than doubt)

} SimOutput_t;

//
// Initializes the output data structure (values to zero, allocate arrays)
//
void init_output(SimOutput_t *output, int population_size)
{
    output->active_fakenews_believers = (int*)malloc(sizeof(int) * 365);
    for (int day = 0; day < 365; ++day) {
    output->active_fakenews_believers[day] = 0;
    }

    output->fakenews_belief_strength = (double*)malloc(sizeof(double) * population_size);

    for (int i=0; i<population_size; ++i) {
    output->fakenews_belief_strength[i] = 0;
    }
}

//
// Destructor
//
void deinit_input(SimInput_t *input)
{
    free(input->contacts_per_day);
    free(input->transmission_probability);
}
void deinit_output(SimOutput_t *output)
{
    free(output->active_fakenews_believers);
    free(output->fakenews_belief_strength);
}


// Init Data with initzial fake news believer
__global__ void cuda_init_Data(int population_size, int starting_fakenews, double *fakenews_belief_strength){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i <  population_size; i += blockDim.x*gridDim.x) {
        fakenews_belief_strength[i] = (i < starting_fakenews) ? 1 : 0;
    }
}

//Compute the number of believers and write into corresoponding array
__global__ void cuda_count_believers(double *cuda_fakenews_believe_strength, int population_size, int *overall_believers){
    __shared__ int shared_believers[num_blocks];

    int believers{0};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < population_size; i += blockDim.x * gridDim.x){
        if (cuda_fakenews_believe_strength[i] > 0.5)
            ++believers;
    }
    shared_believers[threadIdx.x] = believers;
    for (int k = blockDim.x/2; k > 0; k/= 2) {
        __syncthreads();
        if(threadIdx.x < k) {
            shared_believers[threadIdx.x] += shared_believers[threadIdx.x + k];
        }
    }
    if(0 == threadIdx.x) overall_believers[blockIdx.x] = shared_believers[0];

    //unfortunately nop atomic add availavble -> do final summation on cpu
    // if (threadIdx.x == 0) atomicAdd(overall_believers, shared_believers[0]);
}

int count_believers(int* believers_per_block, int size) {
    int tmp_sum{0};
    for (int i = 0; i < size; ++i) {
        tmp_sum += believers_per_block[i];
    }
    return tmp_sum;
}


void run_simulation(const SimInput_t *input, SimOutput_t *output) {

    //Alloc Memory on the GPU
    double *cuda_fakenews_believe_strength;
    cudaMalloc(&cuda_fakenews_believe_strength, sizeof(double)*input->population_size);

    double * cuda_contacts_per_day;
    cudaMalloc(&cuda_contacts_per_day, sizeof(double)*365);

    double *cuda_transmission_probability;
    cudaMalloc(&cuda_transmission_probability, sizeof(double)*365);

    double *cuda_recovery_rate;
    cudaMalloc(&cuda_recovery_rate, sizeof(double) * 365);

    int *cuda_fakenews_believers_per_day;
    cudaMalloc(&cuda_fakenews_believers_per_day, sizeof(int)*365);

    int *cuda_fakenews_believers_per_block;
    cudaMalloc(&cuda_fakenews_believers_per_block, sizeof(int) * num_blocks);

    //Init fakenews believers
    cuda_init_Data<<<num_blocks, num_threads_per_block >>>(input->population_size, input->starting_fakenews, cuda_fakenews_believe_strength);


    double *strength_CPU = new double[input->population_size];
    // cudaMemcpy(strength_CPU, cuda_fakenews_believe_strength, sizeof(double)*input->population_size, cudaMemcpyDeviceToHost);
    // for(int i = 0; i < input->population_size; ++i)
    //     printf("%f\n", strength_CPU[i]);

    int believers_today{0};
    int *fakenews_believers_per_block = new int[num_blocks];
    cuda_count_believers<<<num_blocks, num_threads_per_block>>>(cuda_fakenews_believe_strength, input->population_size, cuda_fakenews_believers_per_block);
    cudaMemcpy(fakenews_believers_per_block, cuda_fakenews_believers_per_block, sizeof(int)*num_blocks, cudaMemcpyDeviceToHost);
    believers_today = count_believers(fakenews_believers_per_block, num_blocks);
    printf("Believers today: %i\n", believers_today);




    delete[] strength_CPU;
    cudaFree(cuda_contacts_per_day);
    cudaFree(cuda_fakenews_believe_strength);
    cudaFree(cuda_fakenews_believers_per_day);
    cudaFree(cuda_recovery_rate);
    cudaFree(cuda_transmission_probability);
}



int main(int argc, char **argv) {
    SimInput_t input;
    SimOutput_t output;
    init_input(&input);
    init_output(&output, input.population_size);
    Timer timer;
    srand(0); // initialize random seed for deterministic output
    timer.reset();
    run_simulation(&input, &output);
    printf("Simulation time: %g\n", timer.get());

    deinit_input(&input);
    deinit_output(&output);

    return EXIT_SUCCESS;
}