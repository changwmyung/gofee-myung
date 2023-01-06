import numpy as np
from abc import ABC, abstractmethod
from ase.data import covalent_radii
from ase.geometry import get_distances
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.geometry import get_distances

from ase.visualize import view

from gofee.candidates.candidate_generation import OffspringOperation


def pos_add_sphere(rattle_strength):
    """Help function for rattling within a sphere
    """
    r = rattle_strength * np.random.rand()**(1/3)
    theta = np.random.uniform(low=0, high=2*np.pi)
    phi = np.random.uniform(low=0, high=np.pi)
    pos_add = r * np.array([np.cos(theta)*np.sin(phi),
                            np.sin(theta)*np.sin(phi),
                            np.cos(phi)])
    return pos_add

def pos_add_sphere_shell(rmin, rmax):
    """Help function for rattling atoms within a spherical
    shell.
    """
    r = np.random.uniform(rmin**3, rmax**3)**(1/3)
    theta = np.random.uniform(low=0, high=2*np.pi)
    phi = np.random.uniform(low=0, high=np.pi)
    pos_add = r * np.array([np.cos(theta)*np.sin(phi),
                            np.sin(theta)*np.sin(phi),
                            np.cos(phi)])
    return pos_add

###
def get_mol_bondlength(mol):
    """
    This funciton is used to check molecule's longest bond length
    """
    Ntop = len(mol)
    num = np.reshape(mol,(Ntop,-1))
    
    r_cov = np.array([covalent_radii[n] for n in num])
    d = np.reshape(r_cov,(1,-1))
    b = np.sort(d)[:, ::-1]
    mol_bl = b[:,0]+b[:,1]
    
    return mol_bl


def get_bondlengths_array(Nslab,n_top,pos_init,pos_final):
    """
    This function is find the atom's distances that composed the molecule(stoichiometry)
    
    pos_init: Atoms object 
        this mean the molecular position before mutation
        ex) pos_init = a
    
    pos_final: Atoms object
        this is molecular position after mutation
        ex) pos_fial = 
    """
    #Nslab=len(slab)
    #Ntop=len(stoichiometry)
    Natoms=Nslab+n_top
    pos_i=pos_init.get_positions()
    pos_f=pos_final.get_positions()
    dist_i=[]
    dist_f=[]
    for i in range(Nslab,Natoms-1):
        dist=get_distances(pos_i[i],pos_i[i+1],cell=pos_init.get_cell(),pbc=pos_init.get_pbc())[1]
        dist_i=np.append(dist_i,dist)
        #print(f'dist_{i},{i+1}={dist}')
    
    for i in range(Nslab,Natoms-1):
        dist=get_distances(pos_f[i],pos_f[i+1],cell=pos_final.get_cell(),pbc=pos_final.get_pbc())[1]
        dist_f=np.append(dist_f,dist)
        #print(f'dist_{i},{i+1}={dist}')
    
    return dist_i, dist_f            


def check_mol_lengths(dist_i,dist_f,bl_limit):
    matrix_size=len(dist_i)
    mol_config=True
    for i in range(matrix_size):
        if abs(dist_i[i]-dist_f[i]) > bl_limit:
            mol_config=False
            return mol_config
            break       
            
        else:
            return mol_config
            continue
            
        return mol_config
###             
###

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


def check_mol_config(ref_mol, mut_mol, bl_limit):
    """
    This function will evaluate the mutated molecule 
    about if the molecule have been broken during the rattle mutation

    If that broken, "mol_config = False"
    """
    ref_bond_list = get_bond_list(ref_mol, bl_limit)
    mut_bond_list = get_bond_list(mut_mol, bl_limit)
    mol_config = True

    if len(ref_bond_list) == len(mut_bond_list):
        for i in range(len(ref_bond_list)):
            if ref_bond_list[i] == mut_bond_list[i]:
                continue
            else:
                mol_config = False
                return mol_config
        return mol_config
    else:
        mol_config = False
        return mol_config
###


class RattleMutation(OffspringOperation):
    """Class to perform rattle mutations on structures.

    Rattles a number of randomly selected atoms within a sphere 
    of radius 'rattle_range' of their original positions.
    
    Parameters:

    n_top: int
        The number of atoms to optimize. Specifically the
        atoms with indices [-n_top:] are optimized.
    
    Nrattle: float
        The average number of atoms to rattle.

    rattle_range: float
        The maximum distance within witch to rattle the
        atoms. Atoms are rattled uniformly within a sphere of this
        radius.

    blmin: float
        The minimum allowed distance between atoms in units of
        the covalent distance between atoms, where d_cov=r_cov_i+r_cov_j.
    
    blmax: float
        The maximum allowed distance, in units of the covalent 
        distance, from a single isolated atom to the closest atom. If
        blmax=None, no constraint is enforced on isolated atoms.

    description: str
        Name of the operation, which will be saved in
        info-dict of structures, on which the operation is applied.    
    """
    def __init__(self, n_top, Nrattle=3, rattle_range=3, bl_limit=0.2,
                 description='RattleMutation', *args, **kwargs):
        OffspringOperation.__init__(self, *args, **kwargs) # rattle_range=3 : default
        self.description = description
        if type(n_top) is int:
            self.n_top = n_top
            self.probability = Nrattle/n_top
        else:
            self.n_top = len(n_top)
            self.molecule = n_top
            self.probability = Nrattle/len(n_top)
        self.rattle_range = rattle_range
        self.bl_limit = bl_limit

    def operation(self, parents):
        a = parents[0]
        a = self.rattle(a)
        return a

    def rattle(self, atoms):
        """ Rattles atoms one at a time within a sphere of radius
        self.rattle_range.
        """
        a = atoms.copy()
        pos_init = atoms.copy()
        Natoms = len(a)
        Nslab = Natoms - self.n_top
        ###
	   # a_atoms_indices = np.array([atom.number for atom in a])
	   # mol = a_atoms_indices[Nslab:]
	   # mol_bl = get_mol_bondlength(mol)

        # Randomly select indices of atoms to permute - in random order.
        indices_to_rattle = np.arange(Nslab,Natoms)[np.random.rand(self.n_top)
                                                     < self.probability]
        indices_to_rattle = np.random.permutation(indices_to_rattle)
        if len(indices_to_rattle) == 0:
            indices_to_rattle = [np.random.randint(Nslab,Natoms)]
        for i in indices_to_rattle:
            posi_0 = np.copy(a.positions[i])
            for _ in range(1000):
		   # while True:
                # Perform rattle
                pos_add = pos_add_sphere(self.rattle_range)
                a.positions[i] += pos_add

                # Check position constraint
                obey_constraint = self.check_constraints(a.positions[i])
                # Check if rattle was valid
                 
				#dist_i, dist_f = get_bondlengths_array(Nslab=Nslab, n_top=self.n_top, pos_init=pos_init, pos_final=a)
                
                #mol_config = check_mol_lengths(dist_i,dist_f,bl_limit)
                #from gofee.candidates import StartGenerator
                mol_config = check_mol_config(ref_mol=self.molecule,mut_mol=a[Nslab:],bl_limit=self.bl_limit)

                valid_operation = mol_config and obey_constraint

                if not valid_operation:
                    a.positions[i] = posi_0
                else:
                    break
       ##         
	   # Perform rattle operations in sequence.
	   # for i in indices_to_rattle:
       #     posi_0 = np.copy(a.positions[i])
       #     for _ in range(200):
       #         # Perform rattle
       #         pos_add = pos_add_sphere(self.rattle_range)
       #         a.positions[i] += pos_add
       #        
       #         # Check position constraint
       #         obey_constraint = self.check_constraints(a.positions[i])
       #         # Check if rattle was valid
       #         valid_bondlengths = self.check_bondlengths(a, indices=[i]) # a

       #         valid_operation = valid_bondlengths and obey_constraint
       #         if not valid_operation:
       #             a.positions[i] = posi_0
       #         else:
       #             break
        if valid_operation:
            return a
        else:
            # If mutation is not successfull in supplied number
            # of trials, return initial structure.
            return None

class RattleMutation2(OffspringOperation):
    """Class to perform rattle mutations on structures.

    Rattles a number of randomly selected atom to the visinity
    of other candomly selected atoms random atom.
    
    Parameters:

    n_top: int
        The number of atoms to optimize. Specifically the
        atoms with indices [-n_top:] are optimized.
    
    Nrattle: float
        The average number of atoms to rattle.

    blmin: float
        The minimum allowed distance between atoms in units of
        the covalent distance between atoms, where d_cov=r_cov_i+r_cov_j.
    
    blmax: float
        The maximum allowed distance, in units of the covalent 
        distance, from a single isolated atom to the closest atom. If
        blmax=None, no constraint is enforced on isolated atoms.

    description: str
        Name of the operation, which will be saved in
        info-dict of structures, on which the operation is applied.    
    """
    def __init__(self, n_top, Nrattle=3, description='RattleMutation',
                 *args, **kwargs):
        OffspringOperation.__init__(self, *args, **kwargs)
        self.description = description
        self.n_top = n_top
        self.probability = Nrattle/n_top

    def operation(self, parents):
        a = parents[0]
        a = self.rattle(a)
        return a

    def rattle(self, atoms):
        """ Repeatedly rattles a random atom to the visinity of another
        random atom.
        """  
        a = atoms.copy()
        Natoms = len(a)
        Nslab = Natoms - self.n_top
        num = a.numbers

        # Randomly select indices of atoms to permute - in random order.
        indices_to_rattle = np.arange(Nslab,Natoms)[np.random.rand(self.n_top)
                                                     < self.probability]
        indices_to_rattle = np.random.permutation(indices_to_rattle)
        if len(indices_to_rattle) == 0:
            indices_to_rattle = [np.random.randint(Nslab,Natoms)]

        # Perform rattle operations in sequence.
        for i in indices_to_rattle:
            posi_0 = np.copy(a.positions[i])
            for _ in range(100):
                j = np.random.randint(Nslab,Natoms)

                # Perform rattle
                covalent_dist_ij = covalent_radii[num[i]] + covalent_radii[num[j]]
                rmin = self.blmin * covalent_dist_ij
                rmax = self.blmax * covalent_dist_ij
                pos_add = pos_add_sphere_shell(rmin, rmax)
                a.positions[i] = np.copy(a.positions[j]) + pos_add

                # Check position constraint 
                obey_constraint = self.check_constraints(a.positions[i])
                # Check if rattle was valid
                valid_bondlengths = self.check_bondlengths(a, indices=[i])
                
                valid_operation = valid_bondlengths and obey_constraint
                if not valid_operation:
                    a.positions[i] = posi_0
                else:
                    break
        if valid_operation:
            return a
        else:
            # If mutation is not successfull in supplied number
            # of trials, return initial structure.
            return None


class PermutationMutation(OffspringOperation):
    """Class to perform permutation mutations on structures.

    Swaps the positions of a number of pairs of unlike atoms.
    
    Parameters:

    n_top: int
        The number of atoms to optimize. Specifically the
        atoms with indices [-n_top:] are optimized.
    
    Npermute: float
        The average number of permutations to perform.
    
    blmin: float
        The minimum allowed distance between atoms in units of
        the covalent distance between atoms, where d_cov=r_cov_i+r_cov_j.
    
    blmax: float
        The maximum allowed distance, in units of the covalent 
        distance, from a single isolated atom to the closest atom. If
        blmax=None, no constraint is enforced on isolated atoms.
    
    description: str
        Name of the operation, which will be saved in
        info-dict of structures, on which the operation is applied.
    """

    def __init__(self, n_top, Npermute=3,
                 description='PermutationMutation', *args, **kwargs):
        OffspringOperation.__init__(self, *args, **kwargs)
        self.description = description
        self.n_top = n_top
        self.probability = Npermute/n_top

    def operation(self, parents):
        a = parents[0]
        a = self.permute(a)
        return a

    def permute(self, atoms):
        """ Permutes atoms of different type in structure.
        """
        a = atoms.copy()
        Natoms = len(a)
        Nslab = Natoms - self.n_top
        num = a.numbers

        # Check if permutation mutation is applicable to structure.
        num_unique_top = list(set(num[-self.n_top:]))
        assert len(num_unique_top) > 1, 'Permutations with one atomic type is not valid'

        # Randomly select indices of atoms to permute - in random order.
        indices_to_permute = np.arange(Nslab,Natoms)[np.random.rand(self.n_top)
                                                   < self.probability]
        indices_to_permute = np.random.permutation(indices_to_permute)
        if len(indices_to_permute) == 0:
            indices_to_permute = [np.random.randint(Nslab,Natoms)]

        # Perform permutations in sequence.
        for i in indices_to_permute:
            for _ in range(100):
                j = np.random.randint(Nslab,Natoms)
                while num[i] == num[j]:
                    j = np.random.randint(Nslab,Natoms)

                # Permute
                pos_i = np.copy(a.positions[i])
                pos_j = np.copy(a.positions[j])
                a.positions[i] = pos_j
                a.positions[j] = pos_i

                # Check if rattle was valid
                valid_bondlengths = self.check_bondlengths(a, indices=[i,j])
                
                if not valid_bondlengths:
                    a.positions[i] = pos_i
                    a.positions[j] = pos_j
                else:
                    break
        if valid_bondlengths:
            return a
        else:
            # If mutation is not successfull in supplied number
            # of trials, return initial structure.
            return None

