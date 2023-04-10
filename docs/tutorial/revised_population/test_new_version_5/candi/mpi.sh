#!/bin/sh
#PBS -V
#PBS -N au50sto
#PBS -q normal 
#PBS -A vasp
#PBS -l select=64:ncpus=32:mpiprocs=32:ompthreads=1
#PBS -l walltime=48:00:00

cd $PBS_O_WORKDIR
mpirun vasp_gam
