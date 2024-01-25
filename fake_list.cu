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
    input->population_size = 8916845;  // Austria's population in 2020 according to Statistik Austria
    // input->population_size = 20000;  // Austria's population in 2020 according to Statistik Austria

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
__global__ void cuda_init_Data(int population_size, int starting_fakenews, double *fakenews_belief_strength){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i <  population_size; i += blockDim.x*gridDim.x) {
        fakenews_belief_strength[i] = (i < starting_fakenews) ? 1 : 0;
    }
}

//Compute the number of believers and write into corresoponding array
__global__ void cuda_count_believers(double *cuda_fakenews_believe_strength, int population_size, int *overall_believers, int *cuda_believers_per_thread){
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shared_believers[num_blocks];

    int believers{0};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < population_size; i += blockDim.x * gridDim.x){
        if (cuda_fakenews_believe_strength[i] > 0.5)
            ++believers;
    }
    cuda_believers_per_thread[tid_global] = believers;
    shared_believers[threadIdx.x] = believers;
    for (int k = blockDim.x/2; k > 0; k/= 2) {
        __syncthreads();
        if(threadIdx.x < k) {
            shared_believers[threadIdx.x] += shared_believers[threadIdx.x + k];
        }
    }
    //workaround due to no atomic add
    if(0 == threadIdx.x) overall_believers[blockIdx.x] = shared_believers[0];

    //unfortunately nop atomic add availavble (threw compilation error) -> do final summation on cpu
    //could achive maybe better performance with atomic add because then the complete days loop
    //could be done in one kernel communicating the array with believers on each day at the end not each day separatly

    // if (threadIdx.x == 0) atomicAdd(overall_believers, shared_believers[0]);
}

__global__ void cuda_pass_on_fakenews(int population_size, double *cuda_rand_values_threads, double *cuda_fakenews_believe_strength, double recovery_rate_today, double transmission_probability_today, double contacts_today, int max_rand_val_per_thread) {
    int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    for (int dude = blockIdx.x * blockDim.x + threadIdx.x; dude < population_size; dude += blockDim.x * gridDim.x){
        cuda_fakenews_believe_strength[dude] -= recovery_rate_today;
        if (cuda_fakenews_believe_strength[dude] > 0.5) {
            for (int contact = 0; contact < contacts_today; ++ contact) {
                double r = cuda_rand_values_threads[tid_global*max_rand_val_per_thread];
                if (r < transmission_probability_today) {
                    r = cuda_rand_values_threads[tid_global*max_rand_val_per_thread+contact+1];
                    int other_person = r*population_size;
                    cuda_fakenews_believe_strength[other_person] = 1;
                }
            }
        }
    }
}


int count_believers(int* believers_per_block, int size) {
    int tmp_sum{0};
    for (int i = 0; i < size; ++i) {
        tmp_sum += believers_per_block[i];
    }
    return tmp_sum;
}

void generate_random_seq(double *array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = ((double)rand()) / (double)RAND_MAX;
    }
}


void run_simulation(const SimInput_t *input, SimOutput_t *output) {
    //get max number of contacts a day to compute array for step 4
    
    double max_contacts_per_day{0};
    for(int i = 0; i < 365; ++i){
        double curr_contacts = input->contacts_per_day[i];
        if (curr_contacts > max_contacts_per_day)
            max_contacts_per_day = curr_contacts;
    }
    printf("max_contacts: %f\n", max_contacts_per_day);
    int max_rand_val_per_thread = (input->population_size + num_blocks * num_threads_per_block -1)/(num_blocks * num_threads_per_block) * 2 * max_contacts_per_day;
    printf("rand vals per block : %i\n", max_rand_val_per_thread);

    //Allocate MEmory on the CPU
    int *fakenews_believers_per_day = new int[356];
    int *believers_per_thread = new int[num_blocks*num_threads_per_block];
    double *tmp_rand_num_array = new double[max_rand_val_per_thread*num_blocks*num_threads_per_block];


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

    int *cuda_believers_per_thread; //needed foir the list of rand numbers
    cudaMalloc(&cuda_believers_per_thread, sizeof(int)*num_blocks*num_threads_per_block);

    double *cuda_rand_values_threads;
    cudaMalloc(&cuda_rand_values_threads, sizeof(double) * num_blocks*num_threads_per_block*max_rand_val_per_thread);

    //Init fakenews believers
    cuda_init_Data<<<num_blocks, num_threads_per_block >>>(input->population_size, input->starting_fakenews, cuda_fakenews_believe_strength);


    double *strength_CPU = new double[input->population_size]; // debugging
    // cudaMemcpy(strength_CPU, cuda_fakenews_believe_strength, sizeof(double)*input->population_size, cudaMemcpyDeviceToHost);
    // for(int i = 0; i < input->population_size; ++i)
    //     printf("%f\n", strength_CPU[i]);


    int num_believers_max{0};
    for (int day = 0; day < 200; ++day) {
        
        // STEP 1: determin number of believers of the day
        int believers_today{0};
        int *fakenews_believers_per_block = new int[num_blocks]; // used for reduction on cpu

        cuda_count_believers<<<num_blocks, num_threads_per_block>>>(cuda_fakenews_believe_strength, input->population_size, cuda_fakenews_believers_per_block, cuda_believers_per_thread);
        cudaMemcpy(fakenews_believers_per_block, cuda_fakenews_believers_per_block, sizeof(int)*num_blocks, cudaMemcpyDeviceToHost);
        believers_today = count_believers(fakenews_believers_per_block, num_blocks);
        fakenews_believers_per_day[day] = believers_today;
        // printf("Believers today: %i\n", believers_today);

        // Step 2:
        // printf("STep2\n");
        int is_pushback{0};
        if (believers_today > num_believers_max) {num_believers_max = believers_today;}
        if (num_believers_max > input->pushback_threshold) {is_pushback = 1;}


        // some diagnostic output
        char pushback[] = " [PUSHBACK]";
        char normal[] = "";
        printf("Day %d%s: %d active fake news believers\n", day, is_pushback ? pushback : normal, believers_today);

        if (believers_today == 0) {
          printf("Fake news pandemic ended on Day %d\n", day);
          break;
        }

        // STEP 3: determin todays transmission/recovery probabilities
        // printf("step3\n");
        double contacts_today = input->contacts_per_day[day];
        double transmission_probability_today = input->transmission_probability[day];
        double recovery_rate_today = input->recovery_rate;

        if (is_pushback) {
            transmission_probability_today /= 5.;
            recovery_rate_today *= 5.;
        }

        //STEP 4: Pass On Fake NEws within thhe population
        //needs to be allocated in the loop because the contacts per day could change each day
        // printf("STep4\n");
        // for (int i = 0; i < num_blocks * num_threads_per_block; ++i) {
        //     generate_random_seq(tmp_rand_num_array, believers_per_thread[i]);
        //     cudaMemcpy(&cuda_rand_values_threads[i*max_rand_val_per_thread], tmp_rand_num_array, sizeof(double)*believers_per_thread[i], cudaMemcpyHostToDevice);
        // }
        generate_random_seq(tmp_rand_num_array, num_blocks*num_threads_per_block*max_rand_val_per_thread);
        cudaMemcpy(cuda_rand_values_threads, tmp_rand_num_array, sizeof(double)*num_blocks*num_threads_per_block*max_rand_val_per_thread, cudaMemcpyHostToDevice);


        // use cuda kernel for computation of populatioin loop
        cuda_pass_on_fakenews<<<num_blocks, num_threads_per_block>>>(input->population_size, cuda_rand_values_threads, cuda_fakenews_believe_strength, recovery_rate_today, transmission_probability_today, contacts_today, max_rand_val_per_thread);
    }

    delete[] strength_CPU;//debugging
    delete[] believers_per_thread;
    delete[] tmp_rand_num_array;
    cudaFree(cuda_contacts_per_day);
    cudaFree(cuda_rand_values_threads);
    cudaFree(cuda_fakenews_believe_strength);
    cudaFree(cuda_fakenews_believers_per_day);
    cudaFree(cuda_recovery_rate);
    cudaFree(cuda_transmission_probability);
    cudaFree(cuda_fakenews_believers_per_block);
    cudaFree(cuda_believers_per_thread);
}



int main(int argc, char **argv) {
    // printf("start main\n");
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