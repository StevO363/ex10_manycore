#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "timer.hpp"



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
    // input->population_sie = 8916845;  // Austria's population in 2020 according to Statistik Austria
    input->population_size = 20;  // Austria's population in 2020 according to Statistik Austria

    input->pushback_threshold   = 20000;   // as soon as we have 20k fake news believers, the general public starts to push back
    input->starting_fakenews    = 10;
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
__global__ void Init_Data(int population_size, int starting_fakenews, double *fakenews_belief_strength){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i <  population_size; i += blockDim.x*gridDim.x) {
        fakenews_belief_strength[i] = (i < starting_fakenews) ? 1 : 0;
    }
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

    //Init fakenews believers
    Init_Data<<<256, 256 >>>(input->population_size, input->starting_fakenews, cuda_fakenews_believe_strength);
    double *strength_CPU = new double[input->population_size];
    cudaMemcpy(strength_CPU, cuda_fakenews_believe_strength, sizeof(double)*input->population_size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < input->population_size; ++i)
        printf("%f\n", strength_CPU[i]);


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