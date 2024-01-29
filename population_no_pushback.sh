#!/bin/bash

out_data1="data/population_GPU_no_pushback.csv"

for i in 50000 100000 500000 1000000 5000000 10000000 50000000 100000000
do
    python3 csmca.py fake_randGPU.cu 10 365 $i >>"$out_data1"
done


# out_data2="data/population_list.csv"

# for i in 100 500 1000 5000 10000 50000 100000 500000
# do
#     python3 csmca.py fake_list.cu 5 $i >>"$out_data2"
# done

# for i in 1000000 5000000 10000000
# do
#     python3 csmca.py fake_list.cu 1 $i >>"$out_data2"
# done
