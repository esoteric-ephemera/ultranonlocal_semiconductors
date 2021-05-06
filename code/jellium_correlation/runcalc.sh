#!/bin/zsh

rm -f tst
rm -rf tst.dSYM

# for ifort
compiler=ifort
compiler_opts=(-g -traceback -i8  -I"${MKLROOT}/include" -O3 -xHost -heap-arrays)
libs=(${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -lpthread -lm -ldl)
# for gfortran
#compiler=gfortran
#compiler_opts='-O2 -march=native'
#libs='-llapack'
subs=(run_eps_c.f90 alda.f90 fxc.f90 mcp07.f90 gki_dlda.f90 eps_c_calc.f90 roots.f90 qv_dlda.f90 gauss_legendre.f90)

$compiler ${compiler_opts[*]} -o tst ${subs[*]} ${libs[*]}

./tst
rm -f tst
rm -rf tst.dSYM

cp jell_eps_c.csv ../data_files/jell_eps_c.csv
