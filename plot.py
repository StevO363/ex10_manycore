import numpy as np
import matplotlib.pyplot as plt
import csv

fake_list = [0.215752, 0.399489, 1.54007, 3.0524, 14.4427, 28.931]
fake_randGPU = [0.0123991, 0.0158201, 0.0195814, 0.0219847, 0.0532599, 0.0866901, 0.391952, 0.776232]

population = [50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]


plt.loglog(population, fake_randGPU, label="RandGPU")
plt.loglog(population[:6], fake_list, label="ListCPU")
plt.grid(True)
plt.legend()
plt.title("Avg Execution Time for Different Population Sizes (with pushback)")
plt.xlabel("Population")
plt.ylabel("Execution Time in Seconds")
plt.savefig("Population.png", dpi=500)
plt.clf()

contacts2 = [0.0572774, 0.187119, 0.325796, 0.463396, 0.600561, 0.7397, 0.881434]
contacts4 = [0.154395, 0.39643, 0.642433, 0.884347, 1.13175, 1.38556, 1.61567]
contacts6 = [0.259552, 0.607445, 0.961678, 1.3122, 1.66392, 2.03011, 2.364]
contacts8 = [0.365007, 0.824449, 1.28175, 1.73965, 2.19664, 2.67341, 3.11432]

days = [50, 100, 150, 200, 250, 300, 350]

plt.plot(days, contacts2, label="2 contacts")
plt.plot(days, contacts4, label="4 contacts")
plt.plot(days, contacts6, label="6 contacts")
plt.plot(days, contacts8, label="8 contacts")
plt.title("Runtimes different number of days")
plt.legend()
plt.xlabel("days")
plt.ylabel("runtimes")
plt.legend()
plt.grid()
plt.savefig("contacts.png", dpi=500)
plt.clf()


runtimes = [0.019713, 0.0258568, 0.0614977, 0.108414, 0.471281, 0.917944]

plt.loglog(population[:6], runtimes, label=" Rand GPU no Pushb")
plt.loglog(population[:6], fake_randGPU[:6], label="Rand GPU with Pushb")
plt.grid(True)
plt.legend()
plt.title("Avg Execution Time for Different Population Sizes (no pushback)")
plt.xlabel("Population")
plt.ylabel("Execution Time in Seconds")
plt.savefig("Population_no_push.png", dpi=500)
plt.clf()
