import numpy as np
import matplotlib.pyplot as plt
import csv

fake_list = [0.215752, 0.399489, 1.54007, 3.0524, 14.4427, 28.931]
fake_randGPU = [0.0123991, 0.0158201, 0.0195814, 0.0219847, 0.0532599, 0.0866901, 0.391952, 0.776232]

population = [50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]


plt.loglog(population, fake_randGPU, label="RandGPU")
plt.loglog(population[:6], fake_list, label="List")
plt.grid(True)
plt.legend()
plt.title("Execution time for different population sizes")
plt.xlabel("Population")
plt.ylabel("Execution Time in Seconds")
plt.savefig("Population.png", dpi=500)