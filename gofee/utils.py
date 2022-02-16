import numpy as np
from abc import ABC, abstractmethod
from ase.data import covalent_radii
from ase.geometry import get_distances
from ase import Atoms

from ase.visualize import view

def check_valid_bondlengths(a, blmin=None, blmax=None, indices=None, check_too_close=True, check_isolated=True):
    """Calculates if the bondlengths between atoms with indices
    in 'indices' and all other atoms are valid. The validity is
    determined by blmin and blmax.

    Parameters:

    a: Atoms object

    blmin: The minimum allowed distance between atoms in units of
    the covalent distance between atoms, where d_cov=r_cov_i+r_cov_j.
    
    blmax: The maximum allowed distance, in units of the covalent 
    distance, from a single isolated atom to the closest atom. If
    blmax=None, no constraint is enforced on isolated atoms.

    indices: The indices of the atoms of which the bondlengths
    with all other atoms is checked. if indices=None all bondlengths
    are checked.
    """
    bl = get_distances_as_fraction_of_covalent(a,indices)
    
    # Filter away self interactions.
    bl = bl[bl > 1e-6].reshape(bl.shape[0],bl.shape[1]-1)
    
    # Check if atoms are too close
    if blmin is not None and check_too_close:
        tc = np.any(bl < blmin)
    else:
        tc = False
        
    # Check if there are isolated atoms
    if blmax is not None and check_isolated:
        isolated = np.any(np.all(bl > blmax, axis=1))
    else:
        isolated = False
        
    is_valid = not tc and not isolated
    return is_valid
    
def get_distances_as_fraction_of_covalent(a, indices=None, covalent_distances=None):
    if indices is None:
        indices = np.arange(len(a))
        
    if covalent_distances is None:
        cd = get_covalent_distance_from_atom_numbers(a, indices=indices)
    else:
        cd = covalent_distances[indices,:]
    _, d = get_distances(a[indices].positions,
                         a.positions,
                         cell=a.get_cell(),
                         pbc=a.get_pbc())
    bl = d/cd
    return bl

def get_covalent_distance_from_atom_numbers(a, indices=None):
    r_cov = np.array([covalent_radii[n] for n in a.get_atomic_numbers()])
    if indices is None:
        r_cov_sub = r_cov
    else:
        r_cov_sub = r_cov[indices]
    cd_mat = r_cov_sub.reshape(-1,1) + r_cov.reshape(1,-1)
    return cd_mat

def get_min_distances_as_fraction_of_covalent(a, indices=None, covalent_distances=None):
    bl = get_distances_as_fraction_of_covalent(a,indices)
    
    # Filter away self interactions.
    bl = bl[bl > 1e-6].reshape(bl.shape[0],bl.shape[1]-1)
    
    return np.min(bl), bl.min(axis=1).argmin()

def array_to_string(arr, unit='', format='0.4f', max_line_length=80):
    msg = ''
    line_length_counter = 0
    for i, x in enumerate(arr):
        string = f'{i} = {x:{format}}{unit},  '
        #string = f"{f'{i}={x:{format}}{unit},':15}"
        line_length_counter += len(string)
        if line_length_counter >= max_line_length:
            msg += '\n'
            line_length_counter = len(string)
        msg += string
    return msg

class OperationConstraint():
    """ Class used to enforce constraints on the positions of
    atoms in mutation and crossover operations.

    Parameters:

    box: Box in which atoms are allowed to be placed. It should
    have the form [] [p0, vspan] where 'p0' is the position of
    the box corner and 'vspan' is a matrix containing the three
    spanning vectors.

    xlim: On the form [xmin, xmax], specifying, in the x-direction, 
    the lower and upper limit of the region atoms can be moved 
    within.

    ylim, zlim: See xlim.
    """
    def __init__(self, box=None, xlim=None, ylim=None, zlim=None):
        self.box = box
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

    def check_if_valid(self, positions):
        """ Returns whether positions are valid under the 
        constraints or not.
        """
        if np.ndim(positions) == 1:
            pos = positions.reshape(-1,3)
        else:
            pos = positions

        if self.box is not None:
            p0, V = self.box
            p_rel = pos - p0  # positions relative to box anchor.
            V_inv = np.linalg.inv(V)
            p_box = p_rel @ V_inv  # positions in box-vector basis.
            if (np.any(p_box < 0) or np.any(p_box > 1)):
                return False
        if self.xlim is not None:
            if (np.any(pos[:,0] < self.xlim[0]) or 
                    np.any(pos[:,0] > self.xlim[1])):
                return False
        if self.ylim is not None:
            if (np.any(pos[:,1] < self.ylim[0]) or 
                    np.any(pos[:,1] > self.ylim[1])):
                return False
        if self.zlim is not None:
            if (np.any(pos[:,2] < self.zlim[0]) or 
                    np.any(pos[:,2] > self.zlim[1])):
                return False

        return True