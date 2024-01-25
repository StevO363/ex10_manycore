

    /**
     * 360.252 - Computational Science on Many-Core Architectures
     * WS 2022/23, TU Wien
     *
     * Simplistic simulator for the spreading of fake news. Inspired by COVID-19 simulations.
     *
     * DISCLAIMER: This simulator is for educational purposes only.
     * It may be arbitrarily inaccurate and should not be used for drawing any conclusions about any actual fake news or virus.
     */

    #include <stdlib.h>
    #include <stdio.h>
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
      input->population_size = 8916845;  // Austria's population in 2020 according to Statistik Austria

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
    // Destructor
    //
    void deinit_input(SimInput_t *input)
    {
      free(input->contacts_per_day);
      free(input->transmission_probability);
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

    void deinit_output(SimOutput_t *output)
    {
      free(output->active_fakenews_believers);
      free(output->fakenews_belief_strength);
    }



    //
    //  Main simulation routine. Steps from day to day, starting at day 0
    //
    void run_simulation(const SimInput_t *input, SimOutput_t *output)
    {
      //
      // Init data. For simplicity we set the first few people to 'fake news spreaders'
      //
      for (int i=0; i<input->population_size; ++i) {
        output->fakenews_belief_strength[i] = (i < input->starting_fakenews) ? 1 : 0;
      }

      //
      // Run simulation by stepping from day to day (at most one year)
      //
      int num_infected_max = 0;
      for (int day=0; day<365; ++day)  // loop over all days of the year
      {
        int is_pushback = 0;

        //
        // Step 1: determine number of infections and recoveries
        //
        int num_infected_current = 0;
        for (int i=0; i<input->population_size; ++i) {

          if (output->fakenews_belief_strength[i] > 0.5)
          {
            num_infected_current += 1;
          }
        }

        output->active_fakenews_believers[day] = num_infected_current;

        //
        // Step 2: Determine whether there is a pushback towards fake news
        //
        if (num_infected_current > num_infected_max)
        {
          num_infected_max = num_infected_current;
        }
        if (num_infected_max > input->pushback_threshold) {
          is_pushback = 1;
        }

        // some diagnostic output
        char pushback[] = " [PUSHBACK]";
        char normal[] = "";
        printf("Day %d%s: %d active fake news believers\n", day, is_pushback ? pushback : normal, num_infected_current);

        if (num_infected_current == 0) {
          printf("Fake news pandemic ended on Day %d\n", day);
          break;
        }

        //
        // Step 3: determine today's transmission and recovery probability
        //
        double contacts_today = input->contacts_per_day[day];
        double transmission_probability_today = input->transmission_probability[day];
        double recovery_rate_today = input->recovery_rate;
        if (is_pushback) { // transmission is reduced with pushback from the general public. Arbitrary factor: 5
          transmission_probability_today /= 5.0;
          recovery_rate_today *= 5.0;
        }


        //
        // Step 4: pass on fake news within population (including recovery)
        //
        for (int i=0; i<input->population_size; ++i) // loop over population
        {
          output->fakenews_belief_strength[i] -= recovery_rate_today;  // believe in fake news fades over time for each person

          if (output->fakenews_belief_strength[i] > 0.5)   // current person is a fake news spreader
          {
            // pass on infection to other persons with transmission probability to each (random) contact
            for (int j=0; j<contacts_today; ++j)
            {
              double r = ((double)rand()) / (double)RAND_MAX;  // random number between 0 and 1
              if (r < transmission_probability_today)
              {
                r = ((double)rand()) / (double)RAND_MAX;       // new random number to determine a random other person to fully convince of fake news
                int other_person = r * input->population_size;

                output->fakenews_belief_strength[other_person] = 1;
              }

            } // for contacts_per_day
          } // if currently infected
        } // for i

      } // for day
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

