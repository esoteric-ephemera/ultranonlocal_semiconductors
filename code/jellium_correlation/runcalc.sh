#!/bin/zsh

rm -f tst
rm -rf tst.dSYM

# use these options for ifort
compiler=ifort
compiler_opts=(-g -traceback -i8  -I"${MKLROOT}/include" -O3 -xHost -heap-arrays -qopenmp)
libs=(${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -lpthread -lm -ldl)
# use these options for gfortran
#compiler=gfortran
#compiler_opts=(-g -fbacktrace -O3 -march=native -fopenmp)
#libs=(-llapack)
subs=(run_eps_c.f90 alda.f90 fxc.f90 mcp07.f90 gki_dlda.f90 eps_c_calc.f90 roots.f90 qv_dlda.f90 gauss_legendre.f90)

$compiler ${compiler_opts[*]} -o tst ${subs[*]} ${libs[*]}

# recommended multiprocessing, set OMP_NUM_THREADS to the
# number of CPUs you want to spread the computations across
OMP_NUM_THREADS=4
export OMP_NUM_THREADS

time ./tst
rm -f tst
rm -rf tst.dSYM

cp jell_eps_c.csv ../data_files/jell_eps_c.csv
