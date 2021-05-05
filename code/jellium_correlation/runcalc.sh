#!/bin/zsh

rm -f tst
rm -rf tst.dSYM

source /opt/intel/mkl/bin/mklvars.sh intel64

# for gfortran
#gfortran -O2 -march=native -o tst alda.f90 fxc.f90 mcp07.f90 gki_dlda.f90  eps_c_calc.f90 test_f_subs.f90 -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib

# for ifort
ifort -g -traceback -O3 -xHost -heap-arrays -o tst run_eps_c.f90 alda.f90 fxc.f90 mcp07.f90 gki_dlda.f90 eps_c_calc.f90  roots.f90 qv_dlda.f90 -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib

./tst
rm -f tst
rm -rf tst.dSYM

cp jell_eps_c.csv ../data_files/jell_eps_c.csv
