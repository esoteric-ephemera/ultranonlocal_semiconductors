import numpy as np
from itertools import product

from constants import pi,Eh_to_eV,bohr_to_ang
density_file = './r2_scan_si_equilibrium_density.csv'
oflnm = density_file.split('.csv')[0]+'_dft.csv'

def get_len(vec):
    return np.sum(vec**2)**(0.5)

def dft_cut(r,fr,rlv,nx,ny,nz,cut):
    xbd = (nx-nx%2)/2
    ybd = (ny-ny%2)/2
    zbd = (nz-nz%2)/2
    gxl = np.arange(-xbd,xbd,1)
    gyl = np.arange(-ybd,ybd,1)
    gzl = np.arange(-zbd,zbd,1)
    fg = np.zeros(0)
    gl = np.zeros((0,3))
    for ivec,gvec in enumerate(product(gxl,gyl,gzl)):
        g = gvec[0]*rlv[0] + gvec[1]*rlv[1] + gvec[2]*rlv[2]
        """ we could use an FFT here, but we only need to use reciprocal lattice vectors below the cutoff  """
        if get_len(g) < cut:
            fac = np.exp(1.j*np.matmul(r,g))
            fg = np.append(fg,np.sum(fr*fac))#/(nx*ny*nz)
            gl = np.vstack((gl,g))
    fg/=fg.shape[0]
    return gl,fg

def dft_n():

    with open('INCAR','r') as infl:
        for ln in infl:
            uln=ln.strip().replace(" ", "").split('=')
            if uln[0].lower()=='encut':
                encut = float(uln[1])
    gcut = (2*encut/Eh_to_eV)**(0.5)

    nion = 1
    with open('CHGCAR','r') as infl:
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
    gv,ng = dft_cut(r,n,rlv,nx,ny,nz,gcut)
    ng *= bohr_to_ang**3

    np.savetxt(oflnm,np.transpose((gv[:,0],gv[:,1],gv[:,2],ng.real,ng.imag)),delimiter=',',header='gx (bohr),gy,gz,Re n(g_vec) (1/bohr**3), Im n(g_vec)')

    return

if __name__=="__main__":

    dft_n()
