from ase.io import write, read
import numpy as np
from abc import ABC, abstractmethod
from ase import Atoms, Atom
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.geometry import get_distances

def get_bond_length(atom1, atom2):
    cov_1 = covalent_radii[atom1.number]
    cov_2 = covalent_radii[atom2.number]
    bond_length = cov_1 + cov_2
    
    return bond_length
    

def get_bond_num_list(slab):
    posi = slab.get_positions()
    bond_num_list = []
    
    for i in range(len(posi)):
        num_of_bond = 0
        for j in range(len(posi)):
            bond_length = get_bond_length(slab[i], slab[j])
            if abs(get_distances(posi[i],posi[j],
                                 cell=slab.get_cell(),
                                 pbc=slab.get_pbc())[1]) < (bond_length + 0.3):
                if abs(get_distances(posi[i],posi[j])[1]) != 0 :
                    num_of_bond += 1
        bond_num_list.append(num_of_bond)
        
    return bond_num_list


def find_defect(slab):  
    #slab = read('POSCAR-template')
    posi = slab.get_positions()
    bond_num_list = get_bond_num_list(slab)
    avg = np.average(bond_num_list)
    defected_posi = []   
    for i in range(len(posi)):
        if bond_num_list[i] < avg:
            defected_posi += [i]
    print(f'Atoms of Defective site = {defected_posi}')
    
    x, y, z, n = 0, 0, 0, 0
   
    for i in defected_posi:
        x+=(posi[i])[0]
        y+=(posi[i])[1]
        n=len(defected_posi)
    x_center = x/n
    y_center = y/n
    center_point = [x_center,y_center]
    print(f'Center Point = {center_point}')
    
    return center_point

