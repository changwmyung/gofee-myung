#!/bin/sh
#PBS -V
#PBS -N austo-8
#PBS -q normal 
#PBS -A vasp
#PBS -l select=64:ncpus=32:mpiprocs=32:ompthreads=1
#PBS -l walltime=48:00:00

module load intel/18.0.3 impi/18.0.3
export CMAKE_INCLUDE_PATH=/apps/compiler/intel/18.0.3/mkl/include
export CMAKE_LIBRARY_PATH=/apps/compiler/intel/18.0.3/mkl/lib/intel64:/apps/compiler/intel/18.0.3/compilers_and_libraries_2018/linux/lib/intel64
export LD_LIBRARY_PATH=$CMAKE_LIBRARY_PATH:$LD_LIBRARY_PATH

cd $PBS_O_WORKDIR
python run.py > log
