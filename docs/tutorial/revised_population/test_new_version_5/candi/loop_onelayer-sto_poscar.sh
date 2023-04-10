rm -r poscar
range=`tail -1 poscar_info.txt`
echo $range

mkdir poscar
cd poscar

for i in $range
do
mkdir poscar-$i
cd poscar-$i
cp ../../mpi.sh ../../INCAR ../../KPOINTS ../../POTCAR ../../POSCAR-$i ../../one_to_threelayer.py ../../POSCAR_added_to_onelayer .
cp POSCAR-$i POSCAR
python one_to_threelayer.py

for PBS_O_WORKDIR in '$PBS_O_WORKDIR'
do
cat >mpi.sh <<!
#!/bin/sh
#PBS -V
#PBS -N sto-$i
#PBS -q normal 
#PBS -A vasp
#PBS -l select=64:ncpus=32:mpiprocs=32:ompthreads=1
#PBS -l walltime=48:00:00

cd $PBS_O_WORKDIR
mpirun vasp_gam
!
done 
qsub mpi.sh
cd ../
done 

cd ../
