import numpy as np
from os import path
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from constants import pi,Eh_to_eV
from discrete_ft_n import oflnm as ft_dens_file
from gauss_quad import gauss_quad
from mcp07 import chi_parser,mcp07_dynamic,gki_dynamic_real_freq

def get_len(vec):
    return np.sum(vec**2)**(0.5)

def init_ang_grid():

    n_quad_pts = 20
    wfile = './grids/gauss_legendre_{:}_pts.csv'.format(n_quad_pts)
    if path.isfile(wfile):
        wg,grid = np.transpose(np.genfromtxt(wfile,delimiter=',',skip_header=1))
    else:
        wg,grid = gauss_quad(n_quad_pts,grid_type='legendre')

    """  phi is the azimuthal angle, grid is cos(theta), with theta the polar angle """
    phi_grid = pi*(grid + 1)
    phi_wg = pi*wg

    """ setting up two-dimensional angular grid for spherical averaging  """
    ang_wg = np.zeros(wg.shape[0]**2)
    q_hat = np.zeros((grid.shape[0]**2,3))
    ind = 0
    for iphi,phi in enumerate(phi_grid):
        for icth,cth in enumerate(grid):
            q_hat[ind,0] = (1-cth**2)**(0.5)*np.cos(phi)
            q_hat[ind,1] = (1-cth**2)**(0.5)*np.sin(phi)
            q_hat[ind,2] = cth
            ang_wg[ind] = phi_wg[iphi]*wg[icth]
            ind += 1
    """ normalization for spherical average is 1/(4 pi)  """
    ang_wg/=(4*pi)
    return q_hat,ang_wg

def wrap_mcp07(q,omega,n):
    dv = {}
    dv['n'] = n
    dv['kF'] = (3*pi*n)**(1/3)
    dv['rs'] = (3/(4*pi*n))**(1/3)
    dv['rsh'] = dv['rs']**(0.5)
    fxc = mcp07_dynamic(q,omega,dv,axis='real',revised=False,param='PZ81')
    return fxc

def wrap_dyn_lda(q,omega,n):
    dv = {}
    dv['n'] = n
    dv['kF'] = (3*pi*n)**(1/3)
    dv['rs'] = (3/(4*pi*n))**(1/3)
    dv['rsh'] = dv['rs']**(0.5)
    #fxc = alda(dv,x_only=False,param='PW92')
    fxc=gki_dynamic_real_freq(dv,omega,x_only=False,revised=False,param='PW92',dimensionless=False)
    return fxc

def calc_alpha(sph_avg=False,fxc='MCP07'):
    """ setting magnitude of |q|<<1 """
    q = 1.e-5
    dftn = np.genfromtxt(ft_dens_file,delimiter=',',skip_header=1)
    g = dftn[:,:3]

    ng0 = np.zeros(dftn.shape[0],dtype='complex')
    ng0.real = dftn[:,3]
    ng0.imag = dftn[:,4]
    """   first find the index of the zero wavevector """
    for iag,ag in enumerate(g):
        if get_len(ag)==0.0:
            """ n(g=0) is the average value of n(r)  """
            n_avg2 = np.abs(ng0[iag])**2
            break
    """ remove the zero-wavevector component from the sum  """
    ng0 = np.delete(ng0,iag,axis=0)
    ng02 = np.abs(ng0)**2
    g = np.delete(g,iag,axis=0)
    gmod = (g[:,0]**2 + g[:,1]**2 + g[:,2]**2)**(0.5)
    Ng = g.shape[0]
    """ only need to evaluate zero-frequency term once   """
    if fxc == 'MCP07':
        fxc_g0 = wrap_mcp07(gmod,np.zeros(Ng),np.abs(ng0))
    elif fxc == 'DLDA':
        fxc_g0 = wrap_dyn_lda(gmod,np.zeros(Ng),np.abs(ng0))

    omega_l = np.arange(0.01,50,0.05)
    alpha = np.zeros(omega_l.shape[0],dtype='complex')

    if sph_avg:
        ofl = './alpha_omega_sph_avg_'+fxc+'.csv'
        """ also only need to evaluate g_vec . q_hat once  """
        q_hat,intwg = init_ang_grid()
        Nq = q_hat.shape[0]
        g_dot_q_hat = np.zeros((Ng,Nq))
        for iag,ag in enumerate(g):
            g_dot_q_hat[iag] = (np.matmul(q_hat,ag))**2
        #n_avg = np.sum(ng0)/ng0.shape[0]
        #ng0 = ng0**2
        for iom,om in enumerate(omega_l):
            if fxc == 'MCP07':
                fxc_g = wrap_mcp07(gmod,om,np.abs(ng0))
            elif fxc == 'DLDA':
                fxc_g = wrap_dyn_lda(gmod,om,np.abs(ng0))
            fxc_diff = fxc_g - fxc_g0
            for iag in range(Ng):
                alpha[iom] += np.sum(intwg*g_dot_q_hat[iag]*fxc_diff[iag]*ng02[iag])/n_avg2
    else:
        ofl = './alpha_omega_'+fxc+'.csv'
        g_dot_q_hat = gmod**2
        intwg = 1.0/3.0
        for iom,om in enumerate(omega_l):
            if fxc == 'MCP07':
                fxc_g = wrap_mcp07(gmod,om,np.abs(ng0))
            elif fxc == 'DLDA':
                fxc_g = wrap_dyn_lda(gmod,om,np.abs(ng0))
            fxc_diff = fxc_g - fxc_g0
            alpha[iom] = intwg*np.sum(g_dot_q_hat*fxc_diff*ng02)/n_avg2


    np.savetxt(ofl,np.transpose((omega_l,alpha.real,alpha.imag)),delimiter=',',header='omega (a.u.), Re alpha(w), Im alpha(w)',fmt='%.18f')

    return

def plotter(sph_avg=False,fxc=[]):

    fig,ax = plt.subplots(2,1,figsize=(8,6))
    max_bd = 0.0
    min_bd = 0.0
    for anfxc in fxc:
        if sph_avg:
            flnm = './alpha_omega_sph_avg_'+anfxc+'.csv'
        else:
            flnm = './alpha_omega_'+anfxc+'.csv'

        om,alp_re,alp_im = np.transpose(np.genfromtxt(flnm,delimiter=',',skip_header=1))
        #om*=Eh_to_eV
        ax[0].plot(om,alp_re,label=anfxc)
        ax[1].plot(om,alp_im)
        max_bd = max([max_bd,alp_re.max()])
        min_bd = min([min_bd,alp_im.min()])
    ax[0].legend(fontsize=14)
    #ax[0].set_yticks(np.arange(0.0,max_bd,.1))
    #ax[1].set_yticks(np.arange(0.0,min_bd,-.1))
    ax[0].set_ylim([0.0,ax[0].get_ylim()[1]])
    ax[1].set_ylim([ax[1].get_ylim()[0],0.0])
    ax[1].set_xlabel('$\\omega$ (a.u.)',fontsize=18)
    ax[0].set_ylabel('$\\mathrm{Re}~\\alpha(\\omega)$',fontsize=18)
    ax[1].set_ylabel('$\\mathrm{Im}~\\alpha(\\omega)$',fontsize=18)
    ax[0].yaxis.set_major_locator(MultipleLocator(.2))
    ax[0].yaxis.set_minor_locator(MultipleLocator(.1))
    ax[1].yaxis.set_major_locator(MultipleLocator(.1))
    ax[1].yaxis.set_minor_locator(MultipleLocator(.05))
    for i in range(2):
        #ax[i].set_xticks(np.arange(0.0,om.max(),10))
        ax[i].set_xlim([0.0,om.max()])
        ax[i].tick_params(axis='both',labelsize=14)
        ax[i].xaxis.set_major_locator(MultipleLocator(10))
        ax[i].xaxis.set_minor_locator(MultipleLocator(5))
    #plt.show()
    plt.savefig('./Si_alpha_omega.png',dpi=600,bbox_inches='tight')
    return

if __name__=="__main__":

    #calc_alpha(sph_avg=False,fxc='MCP07')
    plotter(sph_avg=False,fxc=['DLDA','MCP07'])
