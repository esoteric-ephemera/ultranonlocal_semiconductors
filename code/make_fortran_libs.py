from os import system
from platform import system as sysid

compiler = 'gfortran'

if compiler == 'gfortran':
    compiler_opts = ' -O2 -march=native'
elif compiler == 'ifort':
    compiler_opts = ' -O2 -xHost'

post_opts = ''
# just some nonsense linker because MacOS Catalina breaks libary dependencies
if sysid() == 'Darwin':
    post_opts = ' -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib'

targs = 'alda.f90 fxc.f90 mcp07.f90 gki_dlda.f90'
targs_o = 'flib/{alda,fxc,mcp07,gki_dlda}.o'

comp_str = compiler + compiler_opts +' -c '+ targs + post_opts
print('===============================================================')
print(('Compiling Fortran libraries on {:} with the following options:'.format(sysid())))
print(('Compiler: {:}').format(compiler))
print(('Compiler options: {:}').format(compiler_opts))
#print(comp_str)
system('cd flib; '+comp_str+' ; cd ..')

print('===============================================================')
print('Making Python compatible functions with f2py')
print('===============================================================')
if compiler == 'ifort':
    wcomp = '--fcompiler=intel --f90exec=/usr/local/bin/ifort'
elif compiler == 'gfortran':
    wcomp = '--fcompiler=gnu95'
f2py_str = 'f2py -c '+ wcomp + ' -m ec_calc ./flib/eps_c_calc.f90 ' + targs_o
#f2py_str += ' -L/usr/local/Cellar/gcc/10.2.0/lib/gcc/10/'
#print(f2py_str)
system(f2py_str)
