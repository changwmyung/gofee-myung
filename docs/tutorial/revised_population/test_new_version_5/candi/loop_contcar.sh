mkdir contcar
range=`tail -1 poscar_info.txt`
cd poscar
for i in $range
do
cd poscar-$i
cp CONTCAR ../../contcar/CONTCAR-$i
cd ../
done

cd ../
