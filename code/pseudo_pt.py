import numpy as np
from os import path,system
from itertools import product

from constants import pi,bohr_to_ang,Eh_to_eV
from mcp07 import alda

def norm(vec,scalar=True):
    if scalar:
        return np.sum(vec**2)**(0.5)
    else:
        return (vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2)**(0.5)

def init_g(rlv,nx,ny,nz,cutoff=None):
    tlen = (2*nx + 1)*(2*ny + 1)*(2*nz + 1)
    gvec = np.zeros((tlen,4))
    facx = np.arange(-nx,nx+1,1)
    facy = np.arange(-ny,ny+1,1)
    facz = np.arange(-nz,nz+1,1)
    for ifac,fac in enumerate(product(facx,facy,facz)):
        a,b,c = fac
        gvec[ifac,:3] = a*rlv[0] + b*rlv[1] + c*rlv[2]
    gvec[:,3] = norm(gvec[:,:3],scalar=False)
    if cutoff is not None:
        mask = (gvec[:,3] > 0.0) & (gvec[:,3]**2/2.0<=cutoff)
    else:
        mask = gvec[:,3] > 0.0
    gvec = gvec[mask]
    return gvec

class Crystal:

    __slots__ = 'scale', 'dlv', 'basis', 'rlv'

    """ Container class for crystal geometry """
    def __init__(self,lp,struc):
        # enter lattice parameter in angstrom
        self.scale = lp/bohr_to_ang
        if struc == 'sc':
            lv = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
        elif struc == 'fcc':
            lv = 0.5*np.asarray([[0,1,1],[1,0,1],[1,1,0]])
        elif struc == 'bcc':
            lv = 0.5*np.asarray([[-1,1,1],[1,-1,1],[1,1,-1]])
        self.dlv = self.scale*np.asarray(lv)
        self.rlv = np.zeros((3,3))
        fac = abs(np.linalg.det(self.dlv))
        for ind in range(3):
            self.rlv[ind] = 2*pi*np.cross(self.dlv[(ind+1)%3],self.dlv[(ind+2)%3])/fac
        return

class Pseudo:

    def __init__(self,solid):

        """
            Psuedopotential structure and parameters ("individual" params. of the erratum) from
            C. Fiolhais, J.P. Perdew, S.Q. Armster, J.M. MacLaren, and M. Brajczewska, Phys. Rev. B 51, 14001 (1995), https://doi.org/10.1103/PhysRevB.51.14001,

            and erratum Phys. Rev. B 53, 13193, https://doi.org/10.1103/PhysRevB.53.13193.

            lattice parameters from J. Sun, M. Marsman, G.I. Csonka, A. Ruzsinszky, P. Hao, Y.-S. Kim, G. Kresse, and J.P. Perdew, Phys. Rev. B 84, 035117 (2011), https://doi.org/10.1103/PhysRevB.84.035117
        """

        if solid == 'Al':
            self.lp = 4.018
            self.geom = 'fcc'
            #self.pp_pars = {'rs':2.07,'z': 3, 'nint': 0.705, 'alpha': 3.573, 'R': 0.317}
            self.pp_pars = {'rs':2.07,'z': 3, 'nint': 0.705, 'alpha': 3.572, 'R': 0.317}
        elif solid == 'Na':
            self.lp = 4.214
            self.geom = 'bcc'
            #self.pp_pars = {'rs': 3.93,'z': 1, 'nint': 0.341, 'alpha': 3.517, 'R': 0.492}
            self.pp_pars = {'rs': 3.93,'z': 1, 'nint': 0.341, 'alpha': 3.499, 'R': 0.494}
        return

    def pp(self,q,pars):
        # Eq. 2.11
        qr2 = (q*pars['R'])**2
        alpha = pars['alpha']
        beta = (alpha**3 - 2*alpha)/(4*(alpha**2 - 1))
        A = 0.5*alpha**2 - alpha*beta
        w = -1/qr2 + 1/(qr2 + alpha**2) + 2*alpha*beta/(qr2 + alpha**2)**2 + 2*A/(qr2 + 1)**2
        return 4*pi*pars['z']*pars['R']**2*w

def density_vars(rs):
    dv = {}
    dv['rs'] = rs
    dv['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs
    dv['n'] = 3.0/(4.0*pi*rs**3)
    dv['rsh'] = rs**(0.5)
    dv['wp0'] = (3/rs**3)**(0.5)
    return dv

def alda_chi(q,rs):

    dv = density_vars(rs)
    y = q/(2*dv['kF'])
    slind = 0.5*np.ones(y.shape)
    smask = np.abs(1 - y) > 1.e-10
    slind[smask] += (1 - y[smask]**2)/(4*y[smask])*np.log(np.abs((1 + y[smask])/(1-y[smask])))
    chi0 = -dv['kF']/pi**2*slind
    eps = 1 - (4*pi/q**2 + alda(dv,x_only=False,param='PW92'))*chi0
    return chi0/eps

def get_n_g(solid,full_output=False):

    psp = Pseudo(solid)
    rs0 = psp.pp_pars['rs']
    geom = Crystal(psp.lp,psp.geom)
    ecut = 800/Eh_to_eV # same 800 eV cutoff used for semiconductors
    g_mesh = init_g(geom.rlv,10,10,10,cutoff=ecut)

    gnorm = g_mesh[:,3]
    vG = psp.pp(gnorm,psp.pp_pars)
    chiG = alda_chi(gnorm,rs0)
    n0 = 3/(4*pi*rs0**3)
    nG = n0/psp.pp_pars['z']*chiG*vG

    wdir = './data_files/'+solid+'/'
    if not path.isdir(wdir):
        system('mkdir -p {:}'.format(wdir))
    if full_output:
        hstring = 'Gx, Gy, Gz, |G_vec| (bohr), n(G) (1/bohr**3), rs={:} bohr'.format(rs0)
        np.savetxt(wdir+'{:}_dens_full.csv'.format(solid),np.transpose((g_mesh[:,0],g_mesh[:,1],g_mesh[:,2],g_mesh[:,3],nG)),delimiter=',',header=hstring)
        return g_mesh,nG,rs0
    else:
        hstring = ' G_vec| (bohr), n(G) (1/bohr**3), rs={:} bohr'.format(rs0)
        np.savetxt(wdir+'{:}_dens.csv'.format(solid),np.transpose((gnorm,nG)),delimiter=',',header=hstring)
        return gnorm,nG,rs0

if __name__=="__main__":

    get_n_g('Na')
