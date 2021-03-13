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
writer_exec=/home/tuf53878/vuk/poscar_writer.py

v0="$(grep 'V0' ./ev_data/sjeos_fit.csv)"
v0=${v0:3}
a0=$(echo "e( l(${v0}*8)/3)" | bc -l)
python3 $writer_exec -symm=ds -a=$a0
mpirun -np $PBS_NP $vasp_exec
