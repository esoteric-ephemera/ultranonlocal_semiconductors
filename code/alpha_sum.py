import numpy as np
from os import path
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from constants import crystal,pi,Eh_to_eV
from discrete_ft_n import oflnm as ft_dens_file
from gauss_quad import gauss_quad
from mcp07 import chi_parser,mcp07_dynamic,gki_dynamic_real_freq
from qian_vignale_fxc import fxc_longitudinal as qv_fxc

bigrange = False

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

def wrap_kernel(q,omega,n,wfxc):
    dv = {}
    dv['n'] = n
    dv['kF'] = (3*pi*n)**(1/3)
    dv['rs'] = (3/(4*pi*n))**(1/3)
    dv['rsh'] = dv['rs']**(0.5)
    if wfxc == 'MCP07':
        fxc = mcp07_dynamic(q,omega,dv,axis='real',revised=False,param='PZ81',no_k=False)
    elif wfxc == 'MCP07_k0':
        fxc = mcp07_dynamic(q,omega,dv,axis='real',revised=False,param='PZ81',no_k=True)
    elif wfxc == 'DLDA':
        fxc=gki_dynamic_real_freq(dv,omega,x_only=False,revised=False,param='PZ81',dimensionless=False)
    elif wfxc == 'QV':
        fxc = qv_fxc(dv,omega)
    else:
        raise ValueError('Unknown XC kernel, ', wfxc)
    return fxc

def calc_alpha(fxcl,sph_avg=False):
    """ setting magnitude of |q|<<1 """
    q = 1.e-5
    dftn = np.genfromtxt(ft_dens_file,delimiter=',',skip_header=1)
    g = dftn[:,:3]

    ng0 = np.zeros(dftn.shape[0],dtype='complex')
    ng0.real = dftn[:,3]
    ng0.imag = dftn[:,4]
    """
    import matplotlib.pyplot as plt
    gmod = (g[:,0]**2 + g[:,1]**2 + g[:,2]**2)**(0.5)
    srt_ind = np.argsort(gmod)
    en = gmod[srt_ind]**2/2*Eh_to_eV
    plt.plot(en,np.abs(ng0[srt_ind].real),label='$|\\mathrm{Re} ~n(\\vec G)|$')
    plt.plot(en,np.abs(ng0[srt_ind].imag),label='$|\\mathrm{Im} ~n(\\vec G)|$')
    #plt.plot(en,np.abs(ng0[srt_ind]),label='$|n(\\vec G)|$')
    plt.xlabel('$E_{\\mathrm{cut}}$ (eV)',fontsize=14)
    plt.legend(fontsize=14)
    plt.xlim([0.0,en.max()])
    plt.yscale('log')
    plt.show()
    exit()
    """


    """   first find the index of the zero wavevector """
    """
    for iag,ag in enumerate(g):
        # using FFT, first entry is (0,0,0)
        if get_len(ag)==0.0:
            # n(g=0) is the average value of n(r)
            n_avg2 = np.abs(ng0[iag])**2
            print(('average density {:} bohr**(-3)').format(n_avg2))
            break
    """
    # using FFT, first entry is (0,0,0)
    n_avg2 = np.abs(ng0[0])**2
    print(('average density {:} bohr**(-3); rs_avg = {:} bohr').format(n_avg2**(0.5),(3.0/(4*pi*n_avg2**(0.5)))**(1/3)))
    """ remove the zero-wavevector component from the sum  """
    ng0 = ng0[1:]#np.delete(ng0,iag,axis=0)
    ng02 = np.abs(ng0)**2
    g = g[1:]#np.delete(g,iag,axis=0)
    gmod = (g[:,0]**2 + g[:,1]**2 + g[:,2]**2)**(0.5)
    if not sph_avg:
        g_dot_q_hat = gmod**2

    Ng = g.shape[0]
    if bigrange:
        omega_l = np.linspace(0.01,400.02,500)/Eh_to_eV
    else:
        if crystal == 'Si':
            ulim = 25.0
        elif crystal == 'C':
            ulim = 25.0
        omega_l = np.linspace(0.01,ulim,500)/Eh_to_eV

    if bigrange:
        addn = '_bigrange'
    else:
        addn = ''

    for fxc in fxcl:

        """ only need to evaluate zero-frequency term once   """
        fxc_g0 = wrap_kernel(gmod,0.0,n_avg2**(0.5),fxc)#np.abs(ng0),fxc)#n_avg2**(0.5)*np.ones(gmod.shape[0]),fxc)#

        alpha = np.zeros(omega_l.shape[0],dtype='complex')

        if sph_avg:
            ofl = './alpha_omega_sph_avg_'+fxc+addn+'.csv'
            """ also only need to evaluate g_vec . q_hat once  """
            q_hat,intwg = init_ang_grid()
            Nq = q_hat.shape[0]
            g_dot_q_hat = np.zeros((Ng,Nq))
            for iag,ag in enumerate(g):
                g_dot_q_hat[iag] = (np.matmul(q_hat,ag))**2
            #n_avg = np.sum(ng0)/ng0.shape[0]
            #ng0 = ng0**2
            for iom,om in enumerate(omega_l):
                fxc_g = wrap_kernel(gmod,om,np.abs(ng0),fxc)
                fxc_diff = fxc_g - fxc_g0
                for iag in range(Ng):
                    alpha[iom] += np.sum(intwg*g_dot_q_hat[iag]*fxc_diff[iag]*ng02[iag])/n_avg2
        else:
            ofl = './alpha_omega_'+fxc+addn+'.csv'
            intwg = 1.0/3.0
            for iom,om in enumerate(omega_l):
                fxc_g = wrap_kernel(gmod,om,n_avg2**(0.5),fxc)#np.abs(ng0),fxc)#
                fxc_diff = fxc_g - fxc_g0
                alpha[iom] = intwg*np.sum(g_dot_q_hat*fxc_diff*ng02)/n_avg2


        np.savetxt(ofl,np.transpose((omega_l,alpha.real,alpha.imag)),delimiter=',',header='omega (a.u.), Re alpha(w), Im alpha(w)',fmt='%.18f')

    return

def plotter(fxcl,sph_avg=False):

    clist=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:olive','tab:gray']
    line_styles=['-','--','-.']

    if bigrange:
        olim = 400
    else:
        if crystal == 'Si':
            olim = 25
        elif crystal == 'C':
            olim = 25
    fig,ax = plt.subplots(2,1,figsize=(8,6))
    max_bd = 0.0
    min_bd = 0.0
    alp_re = {}
    alp_im = {}
    om = {}
    if bigrange:
        addn = '_bigrange'
    else:
        addn = ''
    for ifxc,anfxc in enumerate(fxcl):
        if sph_avg:
            flnm = './alpha_omega_sph_avg_'+anfxc+addn+'.csv'
        else:
            flnm = './alpha_omega_'+anfxc+addn+'.csv'

        om[anfxc],alp_re[anfxc],alp_im[anfxc] = np.transpose(np.genfromtxt(flnm,delimiter=',',skip_header=1))
        om[anfxc]*=Eh_to_eV
        mask = om[anfxc]<=olim
        om[anfxc] = om[anfxc][mask]
        alp_re[anfxc] = alp_re[anfxc][mask]
        alp_im[anfxc] = alp_im[anfxc][mask]

        ax[0].plot(om[anfxc],alp_re[anfxc],color=clist[ifxc],linestyle=line_styles[ifxc%len(line_styles)])
        ax[1].plot(om[anfxc],alp_im[anfxc],color=clist[ifxc],linestyle=line_styles[ifxc%len(line_styles)])
        max_bd = max([max_bd,alp_re[anfxc].max()])
        min_bd = min([min_bd,alp_im[anfxc].min()])

    if 'QV' in fxcl:
        ax[0].set_ylim([1.1*alp_re['QV'].min(),1.05*max_bd])
        ax[1].set_ylim([1.05*min_bd,0.0])
        #ax[0].hlines(0.0,ax[0].get_xlim()[0],ax[0].xlim()[1],linestyle='-',color='gray')
    else:
        ax[0].set_ylim([0.0,1.05*max_bd])#ax[0].get_ylim()[1]])
        ax[1].set_ylim([1.05*min_bd,0.0])
    ax[1].set_xlabel('$\\omega$ (eV)',fontsize=16)
    ax[0].set_ylabel('$\\mathrm{Re}~\\alpha(\\omega)$',fontsize=16)
    ax[1].set_ylabel('$\\mathrm{Im}~\\alpha(\\omega)$',fontsize=16)
    if crystal == 'Si':
        ax[0].yaxis.set_major_locator(MultipleLocator(.5))
        ax[0].yaxis.set_minor_locator(MultipleLocator(.25))
        ax[1].yaxis.set_major_locator(MultipleLocator(.2))
        ax[1].yaxis.set_minor_locator(MultipleLocator(.1))
    elif crystal == 'C':
        ax[0].yaxis.set_major_locator(MultipleLocator(.2))
        ax[0].yaxis.set_minor_locator(MultipleLocator(.1))
        ax[1].yaxis.set_major_locator(MultipleLocator(.2))
        ax[1].yaxis.set_minor_locator(MultipleLocator(.1))
    for i in range(2):
        ax[i].set_xlim([0.0,olim])#om.max()])
        ax[i].tick_params(axis='both',labelsize=14)
        if bigrange:
            ax[i].xaxis.set_major_locator(MultipleLocator(50))
            ax[i].xaxis.set_minor_locator(MultipleLocator(25))
        else:
            ax[i].xaxis.set_major_locator(MultipleLocator(2))
            ax[i].xaxis.set_minor_locator(MultipleLocator(1))
    ax[0].xaxis.set_ticklabels([' ' for i in ax[0].xaxis.get_major_ticks()])
    ax[0].tick_params(axis='y',labelsize=14)
    plt.suptitle('{:}, r$^2$SCAN density'.format(crystal),fontsize=16)
    for ifxc,anfxc in enumerate(fxcl):
        if crystal == 'Si':
            offset = 0.06
        elif crystal == 'C':
            offset = 0.03
        if anfxc == 'DLDA':
            lbl = 'Dynamic LDA'
            if bigrange:
                offset = -0.35
        elif anfxc == 'MCP07_k0':
            lbl = 'MCP07, $\\bar{k}=0$'
            if crystal == 'C':
                offset = -0.08
        else:
            lbl = anfxc
            if crystal == 'Si':
                offset = 0.04
            elif crystal == 'C':
                offset = 0.02
        wind = np.argmin(np.abs(om[anfxc] - 0.6*olim))
        p2 = ax[0].transData.transform_point((om[anfxc][wind+1],alp_re[anfxc][wind+1]))
        p1 = ax[0].transData.transform_point((om[anfxc][wind-1],alp_re[anfxc][wind-1]))
        angle = 180/pi*np.arctan((p2[1]-p1[1])/(p2[0]-p1[0]))

        ax[0].annotate(lbl,(0.6*olim,alp_re[anfxc][wind]+offset),color=clist[ifxc],fontsize=12,rotation=angle)
    #plt.show()
    #exit()
    plt.savefig('./{:}_alpha_omega'.format(crystal)+addn+'.pdf',dpi=600,bbox_inches='tight')
    return

if __name__=="__main__":

    calc_alpha(['DLDA','MCP07_k0','MCP07'],sph_avg=False)
    if crystal == 'Si':
        plotter(['MCP07','MCP07_k0','DLDA','QV'],sph_avg=False)
    else:
        plotter(['MCP07','MCP07_k0','DLDA'],sph_avg=False)
