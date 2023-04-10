from ase import Atoms
from ase.visualize import view
from ase.io import read, write, Trajectory

from candidate_filter import CandidateFilter 



candidates = CandidateFilter(structures='structures.traj',
                             stoichiometry=2,
                             energy_gap=1,
                             frequency_measure=10)
candidates.candidate_filter()


