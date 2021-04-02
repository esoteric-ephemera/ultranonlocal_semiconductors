import numpy as np
from itertools import product
from os import path,system

from constants import crystal,pi,Eh_to_eV,bohr_to_ang

if crystal == 'C':
    prefix = '/Users/aaronkaplan/Dropbox/phd.nosync/tddft_ultranonlocality/vasp/C/r2scan/'
    density_file = 'C_r2scan_density.csv'
elif crystal == 'Si':
    prefix = '/Users/aaronkaplan/Dropbox/phd.nosync/tddft_ultranonlocality/vasp/Si/r2scan/'
    density_file = 'r2_scan_si_equilibrium_density.csv'
wdir = './data_files/'+crystal+'/'
oflnm = wdir + density_file.split('.csv')[0]+'_dft.csv'
density_file = prefix + density_file

if not path.isdir(wdir):
    system('mkdir -p {:}'.format(wdir))

def get_len(vec,scalar=True):
    if scalar:
        return np.sum(vec**2)**(0.5)
    else:
        return (vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2)**(0.5)

def dft_cut(r,ifr,rlv,nx,ny,nz,cut):
    fr = np.reshape(ifr,(nx,ny,nz),order='F')
    fg = np.fft.ifftn(fr) # want same phase convention as in VASP
    fg = np.reshape(fg,(nx*ny*nz),order='F')
    inds_x = nx*np.fft.fftfreq(nx)
    if ny == nx:
        inds_y = inds_x
    else:
        inds_y = ny*np.fft.fftfreq(ny)
    if nz == nx:
        inds_z = inds_x
    elif nz == ny:
        inds_z = inds_y
    else:
        inds_z = nz*np.fft.fftfreq(nz)
    gl = np.zeros((nx*ny*nz,3))
    for ivec,gvec in enumerate(product(inds_x,inds_y,inds_z)):
        gl[ivec] = gvec[0]*rlv[0] + gvec[1]*rlv[1] + gvec[2]*rlv[2]
    mask = get_len(gl,scalar=False) < cut
    gl = gl[mask]
    fg = fg[mask]
    return gl,fg

def dft_n():

    with open(prefix+'INCAR','r') as infl:
        for ln in infl:
            uln=ln.strip().replace(" ", "").split('=')
            if uln[0].lower()=='encut':
                encut = float(uln[1])
    gcut = (2*encut/Eh_to_eV)**(0.5)

    nion = 1
    with open(prefix+'CHGCAR','r') as infl:
        for iln,ln in enumerate(infl):
            uln = ln.strip()
            if iln == 1:
                lp = float(uln)/bohr_to_ang
            elif iln == 6:
                nion_l = [int(y) for y in uln.split()]
                nion = int(np.sum(nion_l))
            elif iln == 9 + nion:
                nvec = [int(y) for y in uln.split()]
                nx = nvec[0]
                ny = nvec[1]
                nz = nvec[2]
                break

    dlv = lp*np.asarray([[0.0,0.5,0.5],[0.5,0.0,0.5],[0.5,0.5,0.0]])
    rlv = np.zeros((3,3))
    fac = abs(np.linalg.det(dlv))
    for ind in range(3):
        rlv[ind] = 2*pi*np.cross(dlv[(ind+1)%3],dlv[(ind+2)%3])/fac

    dens = np.genfromtxt(density_file,delimiter=',',skip_header=1)
    r = dens[:,:3]/bohr_to_ang
    n = dens[:,3]
    gv,ng = dft_cut(r,n,rlv,nx,ny,nz,2*gcut)
    ng *= bohr_to_ang**3
    np.savetxt(oflnm,np.transpose((gv[:,0],gv[:,1],gv[:,2],ng.real,ng.imag)),delimiter=',',header='gx (bohr),gy,gz,Re n(g_vec) (1/bohr**3), Im n(g_vec)')

    return

if __name__=="__main__":

    dft_n()
