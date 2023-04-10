touch free_ene.txt
range=`tail -1 poscar_info.txt`
echo $range
cd poscar

for i in $range
do
cd poscar-$i
mv ../../free_ene.txt .
grep free OUTCAR > free.txt
echo 'energy of poscar-'$i >> free_ene.txt
tail -1 free.txt >> free_ene.txt
echo " " >> free_ene.txt
mv free_ene.txt ../../.

cd ../
done

cd ../
