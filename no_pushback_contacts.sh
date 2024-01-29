# !/bin/bash

out_data="data/no_push_contacts2.csv"
out_data2="data/no_push_contacts4.csv"
out_data3="data/no_push_contacts6.csv"
out_data4="data/no_push_contacts8.csv"

for days in 350 400 # 50 100 150 200 250 300 
do
    python3 csmca.py fake_randGPU.cu 5 $days 10000000 2 >> "$out_data"
    python3 csmca.py fake_randGPU.cu 5 $days 10000000 4 >> "$out_data2"
    python3 csmca.py fake_randGPU.cu 5 $days 10000000 6 >> "$out_data3"
    python3 csmca.py fake_randGPU.cu 5 $days 10000000 8 >> "$out_data4"
done