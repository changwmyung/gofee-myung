#!/bin/sh
#PBS -V
#PBS -N gofee
#PBS -q debug 
#PBS -A vasp
#PBS -l select=1:ncpus=64:mpiprocs=64:ompthreads=1
#PBS -l walltime=12:00:00

export OMP_NUM_THREADS=1
export TMI_CONFIG=/apps/compiler/intel/18.0.3/impi/2018.3.222/etc64/tmi.conf
unset I_MPI_FABRICS
export I_MPI_FABRICS_LIST=tmi,tcp
export I_MPI_FALLBACK=1

cd $PBS_O_WORKDIR
#module load intel/18.0.3 impi/18.0.3
#mpirun vasp_std > log
#python md.py
#./run_2.sh
python run.py > log
