import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms

class Population():

    def __init__(self, population_size, gpr, similarity2equal=0.9999):
        self.pop_size = population_size
        self.gpr = gpr
        self.similarity2equal = similarity2equal
        self.largest_energy = np.inf
        self.pop = []
        self.pop_MLrelaxed = []

    def __sort(self):
        E_pop = np.array([a.get_potential_energy() for a in self.pop])
        sorted_indices = np.argsort(E_pop)  # sort from lowest to highest
        self.pop = [self.pop[i] for i in sorted_indices]

    def __save_FandE(self, anew, E, F):
        a = anew.copy()
        results = {'energy': E, 'forces': F}
        calc = SinglePointCalculator(a, **results)
        a.set_calculator(calc)
        return a

    def remove_duplicate_structures(self):
        structures2remove = []
        for i, ai in enumerate(self.pop_MLrelaxed):
            for j, aj in enumerate(self.pop_MLrelaxed):
                if j <= i:
                    continue
                if self.looks_like(ai, aj):
                    # Always remove j index because it will be the one with the largest
                    # energy, since the population is sorted after energy.
                    if not j in structures2remove:
                        structures2remove.append(j)

        # Remove from the end of the population, so indices do not change
        for k in sorted(structures2remove)[::-1]:
            del self.pop[k]
            del self.pop_MLrelaxed[k]

    def add(self, structures):
        if isinstance(structures, Atoms):
            self.__add_structure(structures)
        elif isinstance(structures, list):
            assert isinstance(structures[0], Atoms)
            for a in structures:
                self.__add_structure(a)
            
    def __add_structure(self, a):
        self.remove_duplicate_structures()

        assert 'energy' in a.get_calculator().results
        E = a.get_potential_energy()
        assert 'forces' in a.get_calculator().results
        F = a.get_forces()
        
        if E > self.largest_energy and len(self.pop) == self.pop_size:
            return

        #a = self.__save_FandE(anew, E, F)
        
        # check for similar structure in pop - starting from the largest energy
        for i, ai in enumerate(self.pop_MLrelaxed[::-1]):
            if self.looks_like(a, ai):
                Ei = self.pop[-(i+1)].get_potential_energy()
                if E < Ei:  # if structure in pop is worse, replace.
                    del self.pop[-(i+1)]
                    self.pop.append(a)
                    
                    # sort and set largest energy
                    self.__sort()
                    self.largest_energy = self.pop[-1].get_potential_energy()
                    return
                else:  # If structure in pop is better, discart new structure.
                    return

        # if no similar structure was found in population
        # just add if population us not full
        if len(self.pop) < self.pop_size:
            self.pop.append(a)
        else:  # replace worst
            del self.pop[-1]
            self.pop.append(a)

        # sort and set largest energy
        self.__sort()
        self.largest_energy = self.pop[-1].get_potential_energy()

    def get_structure(self):
        t = np.random.randint(len(self.pop))
        return self.pop[t].copy()

    def get_structure_pair(self):
        t1, t2 = np.random.permutation(len(self.pop))[:2]
        #t1 = np.random.permutation(len(self.pop))[0]
        #t2 = np.random.permutation(len(self.pop))[0]
        structure_pair = [self.pop[t1].copy(), self.pop[t2].copy()]
        return structure_pair

    def looks_like(self, a1, a2):
        f1 = self.gpr.descriptor.get_feature(a1)
        f2 = self.gpr.descriptor.get_feature(a2)
        K0 = self.gpr.K0
        similarity = self.gpr.kernel.kernel_value(f1,f2) / K0
        if similarity > self.similarity2equal:
            return True
        else:
            return False
