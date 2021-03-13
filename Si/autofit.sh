#!/bin/bash
#PBS -l walltime=6:00:00
#PBS -q normal
#PBS -l nodes=2:ppn=20
#PBS -N sjeos_auto_fit
#PBS -j oe
#PBS -o oe.txt
#PBS -m abe
#PBS -M kaplan@temple.edu

module load intel-libs
cd "$PBS_O_WORKDIR"

vasp_exec=/home/tuf53878/vasp6/bin/vasp_std
sjeos_exec=/home/tuf53878/vuk/main.py
writer_exec=/home/tuf53878/vuk/poscar_writer.py
mkdir -p ev_data

a0=5.43

for delta in 0.05 0.01 0.005
do
  for step in 0.0 -1.0 1.0 -2.0 2.0 -3.0 3.0 -4.0 4.0 -5.0 5.0
  do
  ta0=$(echo "${a0}+$step*$delta" | bc)
  python3 $writer_exec -symm=ds -a=$ta0
  mpirun -np $PBS_NP $vasp_exec
  mv OSZICAR "./ev_data/osz_$ta0.txt"
  mv DOSCAR "./ev_data/dos_$ta0.txt"
  mv OUTCAR "./ev_data/out_$ta0.txt"
  done
  cd ev_data ; python3 $sjeos_exec -sjeos ; cd ..
  v0="$(grep 'V0' ./ev_data/sjeos_fit.csv)"
  v0=${v0:3}
  a0=$(echo "e( l(${v0}*8)/3)" | bc -l)
  echo $a0
done
