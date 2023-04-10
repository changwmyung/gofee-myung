import numpy as np

from ase import Atoms
from ase.visualize import view
from ase.io import read, write, Trajectory 
from ase.data.pubchem import pubchem_atoms_search, pubchem_atoms_conformer_search 
from ase.geometry import get_distances
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii

import matplotlib.pyplot as plt 

import linecache
import sys

from time import time


def filter_out(reference, structure, flexibility):
    """
    This function is used to determine whether 
    the candidate structure has the same shape as the reference structure.
    """
    ref_pos = reference.get_positions()
    str_pos = structure.get_positions()
    #print(f'ref_pos={ref_pos}')
    #print(f'str_pos={str_pos}')
    same = True
    for i in range(len(ref_pos)):
        distance = np.linalg.norm(ref_pos[i]-str_pos[i])
        if distance <= 0.05 + flexibility:
            continue
        else:
            same = False
            return same
            break
    return same


def get_bond_list(mol, bl_limit):
    """ 
    This function will find the molecule's bonds and make a bond_list
    """
    Ntop = len(mol)
    bond_list = []
    for i in range(Ntop):
        for j in range(Ntop):
            if i < j:
                i_cov = covalent_radii[mol[i].number]
                j_cov = covalent_radii[mol[j].number]
                if abs(get_distances(mol.get_positions()[i],mol.get_positions()[j])[1]) <= (i_cov + j_cov + bl_limit):
                    #print(f'{i}-{j}: {mol.symbols[i]}-{mol.symbols[j]} bond')
                    bond_list.append(f'{mol.symbols[i]}{i}-{mol.symbols[j]}{j}')
    return bond_list


def check_mol_config(reference_mol, candidate_mol, bl_limit):
    """
    This function will evaluate the mutated molecule 
    about if the molecule have been broken during the rattle mutation
    If that broken, "mol_config = False"
    """
    ref_bond_list = get_bond_list(reference_mol, bl_limit)
    can_bond_list = get_bond_list(candidate_mol, bl_limit)
    #print(f'ref bond list = {ref_bond_list}')
    #print(f'mut bond list = {can_bond_list}')
    mol_config = True

    if len(ref_bond_list) == len(can_bond_list):
        for i in range(len(ref_bond_list)):
            if ref_bond_list[i] == can_bond_list[i]:
                continue
            else:
                mol_config = False
                return mol_config
        return mol_config
    else:
        mol_config = False
        return mol_config 

    
def remove_selec_dynamics(filename):
    read_pos = filename.copy()
    for i in range(len(read_pos)-1):
        if read_pos[i]=='Selective dynamics\n':
           # print(f"REMOVED THE TEXT IN {i+1}'TH LINE; {read_pos[i]}")
            del read_pos[i]
        elif '   F' in read_pos[i]:
           # print(f"REMOVED THE TEXT IN {i+1}'TH LINE; {read_pos[i][60:73]}")
            read_pos[i] = read_pos[i].replace("   F","")
        elif '   T' in read_pos[i]:
           # print(f"REMOVED THE TEXT IN {i+1}'TH LINE; {read_pos[i][60:73]}")
            read_pos[i] = read_pos[i].replace("   T","")
    return read_pos


def poscar_editor(filename):
   # print(f"\n### START TO EDIT {filename} FILE ###")
    with open(filename,'r') as poscar:
        read_pos = poscar.readlines()
        changed_pos = remove_selec_dynamics(read_pos)

        with open(filename,'w') as sys.stdout:
            for i in changed_pos:
                print(i.strip('\n'))

    
class CandidateFilter():
    """
    Method to filter out candidates from global optimized structures.
    
    # Basic setting is like below:
    
            candidates = CandidateFilter(structures='structures.traj')
            candidates.candidate_filter()
        
    1) structures: str; trajectory file's name
    
        
    # Optional setting parameter is that
        
    2) stoichiometry: list of int; atomic number
            ex) CO2 is "[6]+2*[8]" or "[6, 8, 8]"
            If this parameter is set, the evaluation is only apply 
            to the these atoms that adsorbed to the slab.
            
    3) molecule: Atoms object
            If this parameter is set, only candidates that have a
            similar configuration to the molecule set by this parameter
            are filtered out.
    
    4) energy_gap: float or int; energy of eV unit
            If this parameter is set, extract all other types of structures
            that exist between the lowest energy and this parameter value.
        
    5) flexibility: float or int
            This parameter is about how much until the difference 
            will consider as the same structure.
            Default is 2.0 angstrom.
    
    6) num_of_candidates: int
            This parameter is about how much candidates will be
            filtered out.
            Default is 5.
            
    """
    def __init__(self, structures, stoichiometry=None,
                 molecule=None, flexibility=None,
                 energy_gap=None, num_of_candidates=None,
                 frequency_measure=None, poscar_edit=True, 
                 *args, **kwargs):
        
        self.stoichiometry = stoichiometry
        self.structures = structures
        self.molecule = molecule
        self.energy_gap = energy_gap
        self.poscar_edit = poscar_edit
        self.frequency_measure = frequency_measure
        
        if flexibility is None:
            self.flexibility = 1.0
        else:
            self.flexibility = flexibility
             
        if num_of_candidates is None:
            self.num_of_candidates = 5
        else:
            self.num_of_candidates = num_of_candidates
            
            
    def find_minimum(self):
        """
        This function is used to find the structure of the lowest energy.
        """
        
        structures = Trajectory(self.structures)
        num_structures = len(structures)
        ene = [i.get_potential_energy() for i in structures]
        ene = np.asarray(ene)
        minene = ene[np.argmin(ene)]

        for j in range(num_structures): 
            if ene[j] == minene:
                write('min_POSCAR-'+str(j),structures[j])
                min_structures = structures[j]
                
        return min_structures
    

    def candidate_filter(self):
        if self.energy_gap is not None:
            print("\n### ENERGY GAP LOOP;                                                                     ###")
            print("### FIND OUT ALL OF THE CANDIDATES IN BETWEEN THE ENERGY GAP BASED ON THE MINIMUM ENERGY ###") 
            if self.stoichiometry is not None:
                print("\n### FASTER LOOP; ONLY COMPARE THE STOICHIOMETRY PART ###")
                Ntop = self.stoichiometry
            elif self.molecule is not None:
                print("\n### MOLECULE LOOP; FIND OUT THE MOLECULE ###")
                Ntop = len(self.molecule)
            else:
                print("\n### BASIC LOOP; THE SIMPLEST WAY ###")
                Ntop = 0
        else:
            print("\n### NUMBER LOOP;                                     ###")
            print("### FIND OUT THE CANDIDATES UNTIL A SPECIFIED NUMBER ###")
            if self.stoichiometry is not None:
                print("\n### FASTER LOOP; ONLY COMPARE THE STOICHIOMETRY PART ###")
                Ntop = self.stoichiometry
            elif self.molecule is not None:
                print("\n### MOLECULE LOOP; FIND OUT THE MOLECULE ###")
                Ntop = len(self.molecule)
            else:
                print("\n### BASIC LOOP; THE SIMPLEST WAY ###")
                Ntop = 0
        
        structures = Trajectory(self.structures)
        ene = [i.get_potential_energy() for i in structures]
        ene = np.asarray(ene)
        minene = ene[np.argmin(ene)]
        
        for i in range(len(ene)):
            dic_ene = {i : ene[i] for i in range(len(ene))}
        ene_sorted = sorted(dic_ene,key=lambda x:dic_ene[x])           
        
        if self.frequency_measure is not None:
            print("\n### FREQUENCY MEASURE LOOP;                                           ###")
            print("### COUNTS THE SIMILAR STRUCTURES' NUMBER                             ###")
            print("### OBTAIN THE MOST FREQUENTLY FOUND STRUCTURES BY A SPECIFIED NUMBER ###")
            t1 = time()
            candi_num_dic = {}
            candi_dic = {}
            measured_list = []
            ref_mol = self.molecule
            for i in range(len(ene_sorted)):
                num = 1
                if ene_sorted[i] not in measured_list:
                    config = True
                    if self.molecule is not None:
                        mol_config = check_mol_config(reference_mol = ref_mol,
                                                      candidate_mol = structures[ene_sorted[i]][-Ntop:],
                                                      bl_limit = self.flexibility/10)
                        config = config and mol_config         
                    if config:
                        candi_num_dic[ene_sorted[i]] = num
                        candi_dic[ene_sorted[i]] = [ene_sorted[i]] 
                        measured_list += [ene_sorted[i]]

                    for j in range(len(ene_sorted)):
                        if i < j:
                            config = filter_out(reference = structures[ene_sorted[i]][-Ntop:], 
                                                structure = structures[ene_sorted[j]][-Ntop:], 
                                                flexibility = self.flexibility/2)
                            if self.molecule is not None:
                                mol_config = check_mol_config(reference_mol = ref_mol,
                                                              candidate_mol = structures[ene_sorted[i]][-Ntop:],
                                                              bl_limit = self.flexibility/10)
                                config = config and mol_config
                            if config:
                                measured_list += [ene_sorted[j]]
                                candi_dic[ene_sorted[i]] += [ene_sorted[j]]
                                num += 1
                                candi_num_dic[ene_sorted[i]] = num
            print(f'\nlength of measured list: {len(measured_list)}\n')
            print(f'candidates_num_dictionary:\n {candi_num_dic}\n')
            print(f'candidates_dictionary:\n {candi_dic}\n') 
            sorted_candi_dic = sorted(candi_num_dic,key=lambda x:candi_num_dic[x],reverse=True)
            print(f'sorted_candi_dic:\n {sorted_candi_dic}\n')

            t2 = time()
            print(f'\n ## time of frequency measure is {t2-t1} ## \n')
            for i in range(self.frequency_measure):
                print(f'{sorted_candi_dic[i]};{ene[sorted_candi_dic[i]]} = {candi_dic[sorted_candi_dic[i]]} = {candi_num_dic[sorted_candi_dic[i]]}')
                write('POSCAR_freq-'+str(sorted_candi_dic[i]),structures[sorted_candi_dic[i]])
            #return candi_dic
        
        t3 = time()
        if self.molecule is not None:
            ref_mol = self.molecule
            for i in range(len(ene_sorted)):
                mol_config = check_mol_config(reference_mol = ref_mol,
                                              candidate_mol = structures[ene_sorted[i]][-Ntop:],
                                              bl_limit = self.flexibility/10)
                if mol_config:
                    candidates_list = [ene_sorted[i]]
                    #print(f'initial candidates list = {candidates_list}')
                    break
        else:
            candidates_list = [ene_sorted[0]]
            #print(f'initial candidates list = {candidates_list}')
            
        ref = structures[candidates_list[0]]    
        out_of_range = False                
        mol_config = True  
        
        for i in range(len(ene_sorted)):
            config = filter_out(reference = ref[-Ntop:], 
                                structure = structures[ene_sorted[i]][-Ntop:], 
                                flexibility = self.flexibility) 
            if self.molecule is not None:
                mol_config = check_mol_config(reference_mol = ref[-Ntop:],
                                              candidate_mol = structures[ene_sorted[i]][-Ntop:],
                                              bl_limit = self.flexibility/10)
    
            if mol_config and not config:
                candidates_list += [ene_sorted[i]] 
                
                list_1 = candidates_list.copy()
                del list_1[-1]
                for j in list_1:
                    config = filter_out(reference = structures[ene_sorted[i]][-Ntop:], 
                                        structure = structures[j][-Ntop:], 
                                        flexibility = self.flexibility)
                    if self.energy_gap is not None:
                        energy_limit = self.energy_gap + ene[ene_sorted[0]]
                        if ene[ene_sorted[i]] > energy_limit:
                            out_of_range = True
                            print(f'minimum energy is {ene[ene_sorted[0]]}')
                            print(f'energy limit is {energy_limit}')
                            print(f"last structure's({ene_sorted[i]})energy is {ene[ene_sorted[i]]}")
                            print(f'out of range is {out_of_range}\n')

                    if config or out_of_range:
                        #print(f'candidates_list_before = {candidates_list}')
                        del candidates_list[-1]
                        #print(f'candidates_list_after = {candidates_list}')
                        break

            if self.energy_gap is not None:
                if out_of_range:
                    break
            else:
                if len(candidates_list) == self.num_of_candidates:
                    break
        print(f'final_candidates_list = {candidates_list} \n')
        t4 = time()
        print(f'\n ## time of candidate filter is {t4-t3} ## \n')
        
        plt.plot(ene)
        plt.xlabel('Steps', color ='black')
        plt.ylabel('Energy', color = 'black')
        plt.plot(ene, color = 'blue')
        plt.savefig('Energy_graph.png')
        
        write('min_POSCAR-'+str(ene_sorted[0]),structures[ene_sorted[0]])
        write('POSCAR-'+str(ene_sorted[0]),structures[ene_sorted[0]])
        for i in candidates_list:
            write('POSCAR-'+str(i),structures[i])
              
        if self.poscar_edit:
            filename = 'min_POSCAR-'+str(ene_sorted[0])
            poscar_editor(filename)   
            filename = 'POSCAR-'+str(ene_sorted[0])
            poscar_editor(filename)                    
            for i in candidates_list:                
                filename = 'POSCAR-'+str(i)
                poscar_editor(filename)               
            if self.frequency_measure is not None:
                for i in range(self.frequency_measure):
                    filename = 'POSCAR_freq-'+str((sorted_candi_dic[i])) 
                    poscar_editor(filename)
        
        # Loop for saving the extracted energy values 
        sys.stdout = open('energy.txt','w')
        print('min_Energy = ',minene)
        for k in candidates_list:
            print('Energy of POSCAR-',str(k),'= ',ene[k])
        sys.stdout.close()

        # Loop for extraction of the POSCAR files
        sys.stdout = open('poscar_info.txt','w')
        for i in candidates_list:
            print(i,end=' ')
        sys.stdout.close()
           
 
