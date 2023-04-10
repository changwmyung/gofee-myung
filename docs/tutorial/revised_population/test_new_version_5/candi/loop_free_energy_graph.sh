touch free_ene_graph.txt
range=`tail -1 poscar_info.txt`
ref_ene=-3153.376352
echo $range
cd poscar
o=1
j=1
for i in $range
do
cd poscar-$i
mv ../../free_ene_graph.txt .
#grep free OUTCAR > free.txt
#tail -1 free.txt > tail.txt
tail -1 OSZICAR > oszi.txt
#ene_str=`cut tail.txt -b31-45`
#ene_str=`cut oszi.txt -b8-22`
ene_str=`cut oszi.txt -b27-42`
#ene=$((ene_str))
echo $j "  " $ene_str "  " $i >> free_ene_graph.txt
mv free_ene_graph.txt ../../.
j=$(($j+$o))
cd ../
done

cd ../

cat >plot_energy_graph.py <<!
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
ref_ene = $ref_ene
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
!
python plot_energy_graph.py



