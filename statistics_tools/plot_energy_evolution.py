#!/usr/bin/env python                                                                                                                         
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.visualize import view

import sys

runs_name = sys.argv[1]
try:
    dE = int(sys.argv[2])
except:
    dE = None

try:
    Nruns2use = int(sys.argv[3])
except:
    Nruns2use = 12

E_all = []
Epred_all = []
Epred_std_all = []
Ebest_all = []
for i in range(Nruns2use):
    print('progress: {}/{}'.format(i+1,Nruns2use))
    try:
        traj = read(runs_name + '/run{}/structures.traj'.format(i), index=':')
        E = np.array([a.get_potential_energy() for a in traj])
        E_all.append(E)
        Epred = []
        Epred_std = []
        for a in traj:
            try:
                Epred_i = a.info['key_value_pairs']['Epred']
                Epred_std_i = a.info['key_value_pairs']['Epred_std']
                Epred.append(Epred_i)
                Epred_std.append(Epred_std_i)
            except Exception as err:
                #print(err)
                Epred.append(np.nan)
                Epred_std.append(np.nan)
        Epred_all.append(np.array(Epred))
        Epred_std_all.append(np.array(Epred_std))
        
        Ebest = np.min(E)
        Ebest_all.append(Ebest)
        print('Ebest={}'.format(Ebest))
    except Exception as error:
        print(error)

ncol = 4
nrow = np.int(np.ceil(Nruns2use / ncol))
width = ncol*5
height = nrow*5
fig, axes = plt.subplots(nrow, ncol, figsize=(width, height))
for i,(E, Epred, Epred_std, Ebest) in enumerate(zip(E_all, Epred_all, Epred_std_all, Ebest_all)):
    print('len(E)', len(E))
    x = np.arange(len(E))
    irow = i // ncol
    icol = i % ncol
    axes[irow,icol].plot(x, Epred, color='steelblue')
    axes[irow,icol].fill_between(x, Epred-Epred_std, Epred+Epred_std, color='steelblue', alpha=0.4)
    axes[irow,icol].plot(x,E,label='run {0:d}\nEbest={1:.3f}'.format(i,Ebest))
    axes[irow,icol].legend(loc='upper right')
    if dE is not None:
        axes[irow,icol].set_ylim([Ebest, Ebest+dE])
plt.savefig('energyEvol_{}.pdf'.format(runs_name), transparent=True)
plt.show()

