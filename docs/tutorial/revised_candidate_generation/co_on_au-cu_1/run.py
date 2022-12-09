"""
This is an example setting that using 
the make_mol_structure function that involve the spherically random generation manner,
and the MakeBox class.
"""
import numpy as np
import torch

from ase import Atoms
from ase.visualize import view
from ase.io import read, write


from torch.nn import Module

from ase.calculators.dftb import Dftb
from ase.build import fcc111,fcc100
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp

from gofee.candidates import CandidateGenerator, RattleMutation, PermutationMutation, StartGenerator
from gofee.utils import OperationConstraint
from gofee import GOFEE
from gofee.candidates import MakeBox

##### You have to set the calculator, molecule, slab, box, and so on parameters #####

##1) Set a calculator
#calc=Vasp(command='vasp_std',xc='PBE',setups='recommended',lreal='Auto',algo='Fast',nelm=200,ediff=1e-4,lwave=False,lcharg=False,ismear=0,ivdw=20,encut=500,sigma=0.01)
calc = EMT()

##2) Set a molecule which will be positioned to a slab
d = 1.1
molecule = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)])

##3) Set the slab, the molecule will be positioned to this slab
#slab = read('CONTCAR-au_on_cu')
slab = fcc111('Cu', size=(7, 7, 1), vacuum=10.0)
slab += Atoms('4Au', positions=[(11.487,6.632,14.), (12.7635,7.369,12.), (10.211,7.369,12.), (11.487,5.158,12.)])
c = FixAtoms(indices=np.arange(len(slab)))
slab.pbc=True
slab.set_constraint(c)

##4) Set the box, you can set this parameter using the MakeBox class
stoichiometry = [6]+[8]

box_1 = MakeBox(stoichiometry=stoichiometry,
               slab=slab,
               specified_atoms=79,
               center_point=None,
               bl_factor=None,
               shirinkage=None)
box = box_1.make_box()
print(f'box={box}')


##5) Set the sg object use the StartGenerator class
"""
If you want to use the method of molecule's random position to the slab, 
you have to set the parameter 'molecule=molecule'

If you want to use the original version, 
you have to set the parameter 'molecule=None' or not set the parameter 'molecule'
"""

radius, spherical_center = box_1.spherical_parameters() # Use the MakeBox class's function

sg = StartGenerator(molecule=molecule,
                    slab=slab, 
                    box=box, 
                    stoichiometry=stoichiometry,
                    center=spherical_center,
                    radius=radius,
                    half=True)
#sg.make_mol_structure()

"""
If you want to skip the spherically random generation manner,
just set like below:

sg = StartGenerator(slab, stoichiometry, box, molecule)

do not need to set other parameters, such as course ##5).
"""

##6) Set the box_constraint, the rattle, and the candidate_generator object

# Set up constraint for rattle-mutation
box_constraint = OperationConstraint(box=box)

# initialize rattle-mutation
n_to_optimize = len(stoichiometry)
rattle = RattleMutation(n_to_optimize, Nrattle=3, rattle_range=4)

candidate_generator = CandidateGenerator([0.2, 0.8], [sg, rattle])



##### Initialize and run search #####
search = GOFEE(calc=calc,
               restart=None,
               startgenerator=sg,
               candidate_generator=candidate_generator,
               max_steps=150,
               population_size=5,
               position_constraint=box_constraint,
               bfgs_traj='bfgs.traj')
""" 
If you set "bfgs_traj='bfgs.traj", then the 'bfgs.traj' and 'bfgs_tot.traj' files will be generated,
but "bfgs_traj=None" or not set this parameter, then the bfgs related trajectory files will not be generated. 

The 'bfgs_tot.traj' is include the assembled information of each local optimization processes.
"""
search.run()

