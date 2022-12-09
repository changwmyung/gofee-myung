import numpy as np
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.ga.utilities import get_mic_distance
from ase.io import read, write
from ase.io.trajectory import Trajectory

from gofee.utils import check_valid_bondlengths

import traceback
import sys


class BFGSLineSearch_constrained(BFGSLineSearch):
    def __init__(self, atoms, pos_init=None, restart=None, logfile='-', maxstep=.2,
                 trajectory=None, c1=0.23, c2=0.46, alpha=10.0, stpmax=50.0,
                 master=None, force_consistent=None,
                 blmin=None, blmax=None, max_relax_dist=4.0, 
                 position_constraint=None, rk=None):
        """
        add maximum displacement of single atoms to BFGSLineSearch:

        max_relax_dist: maximum distance the atom is alowed to move from it's initial position.
        in units of it's covalent distance.
        """

        self.rk = rk  # for testing
        
        self.blmin = blmin
        self.blmax = blmax
        self.position_constraint=position_constraint
        
        self.max_relax_dist = max_relax_dist
        if pos_init is not None:
            self.pos_init = pos_init
        else:
            self.pos_init = np.copy(atoms.positions)

        self.cell = atoms.get_cell()
        self.pbc = atoms.get_pbc()

        BFGSLineSearch.__init__(self, atoms, restart=restart, logfile=logfile, maxstep=maxstep,
                                trajectory=trajectory, c1=c1, c2=c2, alpha=alpha, stpmax=stpmax,
                                master=master, force_consistent=force_consistent)

    def converged(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self.atoms.get_forces()
        if hasattr(self.atoms, 'get_curvature'):
            return ((forces**2).sum(axis=1).max() < self.fmax**2 and
                    self.atoms.get_curvature() < 0.0)

        # Check constraints
        terminate_due_to_constraints = self.check_constraints()
        if terminate_due_to_constraints:
            return True
        
        return (forces**2).sum(axis=1).max() < self.fmax**2

    def check_constraints(self):
        # Check if stop due to large displacement
        valid_displace = self.check_displacement()
        # Check if stop due to position-constraint
        valid_pos = self.check_positions()
        # Check if stop due to invalid bondlengths
        valid_bondlengths = self.check_bondlengths()
        
        if not valid_displace or not valid_pos or not valid_bondlengths:
            #self.r0 = self.atoms_prior.get_positions()
            return True
    
    def check_displacement(self):
        valid_displace = True
        if self.max_relax_dist is not None:
            d_relax = np.array([get_mic_distance(p1,p2,self.cell,self.pbc) 
                                for p1,p2 in zip(self.pos_init,self.atoms.get_positions())])
            if np.any(d_relax > self.max_relax_dist):
                valid_displace = False
        return valid_displace

    def check_positions(self):
        valid_pos = True
        if self.position_constraint is not None:
            # get indices of non-fixed atoms
            indices = np.arange(self.atoms.get_number_of_atoms())
            for constraint in self.atoms.constraints:
                if isinstance(constraint, FixAtoms):
                    indices_fixed = constraint.get_indices()
                    indices_not_fixed = np.delete(np.arange(self.atoms.get_number_of_atoms()), indices_fixed)
            pos_not_fixed = self.atoms.positions[indices_not_fixed]
            valid_pos = self.position_constraint.check_if_valid(pos_not_fixed)
        return valid_pos

    def check_bondlengths(self):
        valid_bondlengths = True
        if self.blmin is not None or self.blmax is not None:
            valid_bondlengths = check_valid_bondlengths(self.atoms, self.blmin, self.blmax)
            if not valid_bondlengths:
                valid_bondlengths = False
        return valid_bondlengths

###
def combine_traj(bfgs_tot='bfgs_tot.traj', bfgs_loc='bfgs_loc.traj'):
    """
    'bfgs_loc.traj' will be added to the 'bfgs_tot.traj'
    
    You will maybe only need to specify the 'bfgs_loc.traj'.  
    """
    
    bfgs_tot = Trajectory(filename=bfgs_tot, mode='a')
    bfgs_loc = Trajectory(filename=bfgs_loc)
    
    for atoms in bfgs_loc:
        bfgs_tot.write(atoms)

    bfgs_tot.close()
    bfgs_loc.close()    
###


def relax(structure, calc, Fmax=0.05, steps_max=200, max_relax_dist=None, position_constraint=None, bfgs_traj=None):
    a = structure.copy()
    # Set calculator 
    a.set_calculator(calc)
    pos_init = a.get_positions()

    # Catch if linesearch fails
    try:
        dyn = BFGSLineSearch_constrained(a,
                                         logfile=None,
                                         pos_init=pos_init,
                                         max_relax_dist=max_relax_dist,
                                         trajectory=bfgs_traj,
                                         position_constraint=position_constraint)

                                        # append_trajectory=True,
        dyn.run(fmax = Fmax, steps = steps_max)
        ###
        if bfgs_traj is None:
            pass
        else:
            combine_traj(bfgs_tot='bfgs_tot.traj', bfgs_loc=bfgs_traj)
        ###
    except Exception as err:
        print('Error in surrogate-relaxation:', err, flush=True)
        traceback.print_exc()
        traceback.print_exc(file=sys.stderr)
    return a
