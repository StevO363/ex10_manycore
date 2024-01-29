#include "timer.hpp"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

const int num_blocks = 256;
const int num_threads_per_block = 256;

//
// Data container for simulation input
//
typedef struct {

  size_t population_size; // Number of people to simulate

  //// Configuration
  int pushback_threshold; // Number of fake news believers required for
                          // mainstream pushback
  int starting_fakenews; // Number of people believing in fake news at the start
                         // of the year
  double recovery_rate;  // Rate of a person giving up on fake news and becoming
                         // reasonable again

  // for each day:
  int *contacts_per_day; // number of other persons met each day to whom the
                         // fake news may be passed on
  double *transmission_probability; // how likely it is to pass on the fake news
                                    // to another person

} SimInput_t;

//
// Constructor for the simulation input structure
//
void init_input(SimInput_t *input, size_t population = 8916845,
                int contacts = 2, int pushback = 20000) {
  input->population_size =
      population; // Austria's population in 2020 according to Statistik Austria
  // input->population_size = 20000;  // Austria's population in 2020 according
  // to Statistik Austria

  input->pushback_threshold =
      pushback; // as soon as we have 20k fake news believers, the general
                // public starts to push back
  input->starting_fakenews = 10;
  input->recovery_rate = 0.01;

  input->contacts_per_day = (int *)malloc(sizeof(int) * 365);
  input->transmission_probability = (double *)malloc(sizeof(double) * 365);
  for (int day = 0; day < 365; ++day) {
    input->contacts_per_day[day] =
        contacts; // arbitrary assumption of six possible transmission contacts
                  // per person per day, all year
    input->transmission_probability[day] =
        0.1; // 10 percent chance of convincing a contact of fake news
  }
}

//
// Data container for simulation output
//
typedef struct {
  // for each day:
  int *active_fakenews_believers; // number of people currently believing in
                                  // fake news

  // for each person:
  double *fakenews_belief_strength; // 0 if completely fake-news absent, 1 if
                                    // fully believing in fake news. A person is
                                    // considered to be fake news convinced if
                                    // this number is larger than 0.5 (i.e. more
                                    // belief than doubt)

} SimOutput_t;

//
// Initializes the output data structure (values to zero, allocate arrays)
//
void init_output(SimOutput_t *output, int population_size) {
  output->active_fakenews_believers = (int *)malloc(sizeof(int) * 365);
  for (int day = 0; day < 365; ++day) {
    output->active_fakenews_believers[day] = 0;
  }

  output->fakenews_belief_strength =
      (double *)malloc(sizeof(double) * population_size);

  for (int i = 0; i < population_size; ++i) {
    output->fakenews_belief_strength[i] = 0;
  }
}

//
// Destructor
//
void deinit_input(SimInput_t *input) {
  free(input->contacts_per_day);
  free(input->transmission_probability);
}
void deinit_output(SimOutput_t *output) {
  free(output->active_fakenews_believers);
  free(output->fakenews_belief_strength);
}

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
  double scaling = 1. / UINT_MAX;
  return scaling * (cuda_taus(z_taus) ^ cuda_LCG(z_LCG));
}

// Init Data with initzial fake news believer
__global__ void cuda_init_Data(int population_size, int starting_fakenews,
                               double *fakenews_belief_strength) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < population_size;
       i += blockDim.x * gridDim.x) {
    fakenews_belief_strength[i] = (i < starting_fakenews) ? 1 : 0;
  }
}

// Compute the number of believers and write into corresoponding array
__global__ void cuda_count_believers(double *cuda_fakenews_believe_strength,
                                     int population_size,
                                     int *overall_believers,
                                     int *cuda_believers_per_thread) {
  int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int shared_believers[num_blocks];

  int believers{0};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < population_size;
       i += blockDim.x * gridDim.x) {
    if (cuda_fakenews_believe_strength[i] > 0.5)
      ++believers;
  }
  cuda_believers_per_thread[tid_global] = believers;
  shared_believers[threadIdx.x] = believers;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_believers[threadIdx.x] += shared_believers[threadIdx.x + k];
    }
  }
  // workaround due to no atomic add
  if (0 == threadIdx.x)
    overall_believers[blockIdx.x] = shared_believers[0];

  // unfortunately no atomic add availavble (threw compilation error) -> do
  // final summation on cpu
  //  if (threadIdx.x == 0) atomicAdd(overall_believers, shared_believers[0]);
}

__global__ void cuda_pass_on_fakenews(int population_size,
                                      double *cuda_fakenews_believe_strength,
                                      double recovery_rate_today,
                                      double transmission_probability_today,
                                      double contacts_today,
                                      unsigned *cuda_states_taus,
                                      unsigned *cuda_states_LCG) {
  int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
  for (int dude = blockIdx.x * blockDim.x + threadIdx.x; dude < population_size;
       dude += blockDim.x * gridDim.x) {
    cuda_fakenews_believe_strength[dude] -= recovery_rate_today;
    if (cuda_fakenews_believe_strength[dude] > 0.5) {
      for (int contact = 0; contact < contacts_today; ++contact) {
        double r = cuda_gen_rand_num(cuda_states_taus[tid_global],
                                     cuda_states_LCG[tid_global]);
        if (r < transmission_probability_today) {
          r = cuda_gen_rand_num(cuda_states_taus[tid_global],
                                cuda_states_LCG[tid_global]);
          int other_person = r * population_size;
          cuda_fakenews_believe_strength[other_person] = 1;
        }
      }
    }
  }
}

int count_believers(int *believers_per_block, int size) {
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

  // Allocate Memory on the CPU
  int *fakenews_believers_per_day = new int[356];
  int *believers_per_thread = new int[num_blocks * num_threads_per_block];
  unsigned *state_init = new unsigned[num_blocks * num_threads_per_block];

  // Alloc Memory on the GPU
  double *cuda_fakenews_believe_strength;
  cudaMalloc(&cuda_fakenews_believe_strength,
             sizeof(double) * input->population_size);

  double *cuda_contacts_per_day;
  cudaMalloc(&cuda_contacts_per_day, sizeof(double) * 365);

  double *cuda_transmission_probability;
  cudaMalloc(&cuda_transmission_probability, sizeof(double) * 365);

  double *cuda_recovery_rate;
  cudaMalloc(&cuda_recovery_rate, sizeof(double) * 365);

  int *cuda_fakenews_believers_per_day;
  cudaMalloc(&cuda_fakenews_believers_per_day, sizeof(int) * 365);

  int *cuda_fakenews_believers_per_block;
  cudaMalloc(&cuda_fakenews_believers_per_block, sizeof(int) * num_blocks);

  int *cuda_believers_per_thread; // needed foir the list of rand numbers
  cudaMalloc(&cuda_believers_per_thread,
             sizeof(int) * num_blocks * num_threads_per_block);

  unsigned *cuda_states_taus, *cuda_states_LCG;
  cudaMalloc(&cuda_states_taus,
             sizeof(unsigned) * num_blocks * num_threads_per_block);
  cudaMalloc(&cuda_states_LCG,
             sizeof(unsigned) * num_blocks * num_threads_per_block);

  // init states for rand generators;
  for (int i = 0; i < num_blocks * num_threads_per_block; ++i) {
    state_init[i] = static_cast<unsigned int>(static_cast<double>(rand()) /
                                              RAND_MAX * UINT_MAX);
  }

  cudaMemcpy(cuda_states_taus, state_init,
             sizeof(unsigned) * num_blocks * num_threads_per_block,
             cudaMemcpyHostToDevice);

  for (int i = 0; i < num_blocks * num_threads_per_block; ++i) {
    state_init[i] = static_cast<unsigned int>(static_cast<double>(rand()) /
                                              RAND_MAX * UINT_MAX);
  }

  cudaMemcpy(cuda_states_LCG, state_init,
             sizeof(unsigned) * num_blocks * num_threads_per_block,
             cudaMemcpyHostToDevice);

  // Init fakenews believers
  cuda_init_Data<<<num_blocks, num_threads_per_block>>>(
      input->population_size, input->starting_fakenews,
      cuda_fakenews_believe_strength);

  int num_believers_max{0};
  for (int day = 0; day < 365; ++day) {

    // STEP 1: determin number of believers of the day
    int believers_today{0};
    int *fakenews_believers_per_block =
        new int[num_blocks]; // used for reduction on cpu

    cuda_count_believers<<<num_blocks, num_threads_per_block>>>(
        cuda_fakenews_believe_strength, input->population_size,
        cuda_fakenews_believers_per_block, cuda_believers_per_thread);
    cudaMemcpy(fakenews_believers_per_block, cuda_fakenews_believers_per_block,
               sizeof(int) * num_blocks, cudaMemcpyDeviceToHost);
    believers_today = count_believers(fakenews_believers_per_block, num_blocks);
    fakenews_believers_per_day[day] = believers_today;

    // Step 2:
    int is_pushback{0};
    if (believers_today > num_believers_max) {
      num_believers_max = believers_today;
    }
    if (num_believers_max > input->pushback_threshold) {
      is_pushback = 1;
    }

    // some diagnostic output
    // char pushback[] = " [PUSHBACK]";
    // char normal[] = "";
    // printf("Day %d%s: %d active fake news believers\n", day, is_pushback ?
    // pushback : normal, believers_today);

    if (believers_today == 0) {
      printf("Fake news pandemic ended on Day %d\n", day);
      break;
    }

    // STEP 3: determin todays transmission/recovery probabilities
    double contacts_today = input->contacts_per_day[day];
    double transmission_probability_today =
        input->transmission_probability[day];
    double recovery_rate_today = input->recovery_rate;

    if (is_pushback) {
      transmission_probability_today /= 5.;
      recovery_rate_today *= 5.;
    }
    // use cuda kernel for computation of populatioin loop
    cuda_pass_on_fakenews<<<num_blocks, num_threads_per_block>>>(
        input->population_size, cuda_fakenews_believe_strength,
        recovery_rate_today, transmission_probability_today, contacts_today,
        cuda_states_taus, cuda_states_LCG);
  }

  delete[] believers_per_thread;
  cudaFree(cuda_contacts_per_day);
  cudaFree(cuda_fakenews_believe_strength);
  cudaFree(cuda_fakenews_believers_per_day);
  cudaFree(cuda_recovery_rate);
  cudaFree(cuda_transmission_probability);
  cudaFree(cuda_fakenews_believers_per_block);
  cudaFree(cuda_believers_per_thread);
}

int main(int argc, char **argv) {
  size_t population = 8916845;
  int contacts = 2;
  int pushback = 20000;
  int runs = 10;

  if (1 < argc)
    runs = std::stoi(argv[1]);
  if (2 < argc)
    population = std::stoll(argv[2]);
  if (3 < argc)
    contacts = std::stoi(argv[3]);
  if (4 < argc)
    pushback = std::stoi(argv[4]);
  Timer timer;
  double time{0};
  printf("Number of runs: %i\npopulation: %e, contacts: %i, pusback: %i\n",
         runs, static_cast<double>(population), contacts, pushback);
  for (int run = 0; run < runs; ++run) {
    SimInput_t input;
    SimOutput_t output;
    init_input(&input, population, contacts, pushback);
    init_output(&output, input.population_size);
    timer.reset();
    run_simulation(&input, &output);
    time += timer.get();
    deinit_input(&input);
    deinit_output(&output);
  }
  printf("avg Simulation time: %g\n", time / runs);
  return EXIT_SUCCESS;
}
