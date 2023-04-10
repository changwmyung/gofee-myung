import numpy as np
import matplotlib.pyplot as plt

plt.figure()
ref_ene = -3153.34261
ene = np.loadtxt('free_ene_graph.txt',dtype='str')
ene_range = ene[:,0].astype(int)
ene_list = ene[:,1].astype(float)
poscar_index = ene[:,2].astype(int)

gap = ene_list-ref_ene
plt.plot(ene_range,gap)
for i in range(len(ene_range)):
    plt.annotate('%i' %(poscar_index[i]),xy=(i,gap[i]))
plt.savefig('Binding_energy_graph')


plt.figure()
ene_before = np.loadtxt('energy.txt',dtype='str',skiprows=1)
ene_b_range = []
ene_b_list = ene_before[:,5].astype(float)
poscar_b_index = ene_before[:,3].astype(int)
for i in range(len(ene_b_list)):
    ene_b_range += [i+1]

plt.plot(ene_b_range,ene_b_list)
for i in range(len(ene_b_list)):
    plt.annotate('%i' %(poscar_b_index[i]),xy=(i,ene_b_list[i]))
plt.savefig('Not_relaxed_Candidates_Energy_graph')
