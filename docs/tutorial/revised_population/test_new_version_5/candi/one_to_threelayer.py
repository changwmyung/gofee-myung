
from ase import Atoms
from ase.visualize import view
from ase.io import read, write

import numpy as np

two_layer = read('POSCAR_added_to_onelayer')
one_layer = read('POSCAR')
three_layer = one_layer + two_layer

write('POSCAR',three_layer,sort=True)
