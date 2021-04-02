import numpy as np
from os import path,mkdir
from itertools import product
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from constants import crystal,pi,Eh_to_eV
#from gauss_quad import gauss_quad
from mcp07 import chi_parser,mcp07_dynamic,gki_dynamic_real_freq
from qian_vignale_fxc import fxc_longitudinal as qv_fxc
from pseudo_pt import get_n_g

metal_regex = ['Al','Na']
semicond_regex = ['Si','C']
if crystal in semicond_regex:
    from discrete_ft_n import dft_n
    from discrete_ft_n import oflnm as ft_dens_file

bigrange = True
use_multiprocesing = True
if not path.isdir('./figs'):
    mkdir('./figs')
wdir = './data_files/{:}/'.format(crystal)

ulim_d = {'Si':25.0, 'C':25.0, 'Al': 250.0, 'Na': 100.0}

def get_len(vec):
    return np.sum(vec**2)**(0.5)

def wrap_kernel(q,omega,n,wfxc):
    dv = {}
    dv['n'] = n
    dv['kF'] = (3*pi**2*n)**(1/3)
    dv['rs'] = (3/(4*pi*n))**(1/3)
    dv['rsh'] = dv['rs']**(0.5)
    dv['wp0'] = (3/dv['rs']**3)**(0.5)
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

def plot_fourier_components():

    if crystal in metal_regex:

        gmod,ng0,rs0 = get_n_g(crystal,full_output=False)
    else:
        if not path.isfile(ft_dens_file):
            dft_n()
        dftn = np.genfromtxt(ft_dens_file,delimiter=',',skip_header=1)
        g = dftn[:,:3]

        ng0 = np.zeros(dftn.shape[0],dtype='complex')
        ng0.real = dftn[:,3]
        ng0.imag = dftn[:,4]
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
    return

def init_n_g(solid):

    if solid in metal_regex:

        gmod,ng0,rs0 = get_n_g(solid,full_output=False)
        n_0_bar = 3/(4*pi*rs0**3)

    elif solid in semicond_regex:
        if not path.isfile(ft_dens_file):
            dft_n()
        dftn = np.genfromtxt(ft_dens_file,delimiter=',',skip_header=1)
        g = dftn[:,:3]

        ng0 = np.zeros(dftn.shape[0],dtype='complex')
        ng0.real = dftn[:,3]
        ng0.imag = dftn[:,4]

        #   first find the index of the zero wavevector
        if get_len(g[0])==0.0:
            # using FFT, first entry is (0,0,0)
            n_0_bar = np.abs(ng0[0])
            # remove the zero-wavevector component from the sum
            ng0 = ng0[1:]
            g = g[1:]
        else:
            for iag,ag in enumerate(g):
                if get_len(ag)==0.0:
                    n_0_bar = np.abs(ng0[iag])
                    ng0 = np.delete(ng0,iag,axis=0)
                    g = np.delete(g,iag,axis=0)
                    break
        gmod = (g[:,0]**2 + g[:,1]**2 + g[:,2]**2)**(0.5)

    return gmod,np.abs(ng0)**2,n_0_bar


def calc_alpha(fxcl,):

    dyn_only_regex = ['DLDA','QV']

    gmod,ng02,n_0_bar = init_n_g(crystal)
    print(('average density {:} bohr**(-3); rs_avg = {:} bohr').format(n_0_bar,(3.0/(4*pi*n_0_bar))**(1/3)))
    n_avg2 = n_0_bar**2
    g_dot_q_hat = gmod**2

    Ng = gmod.shape[0]
    if bigrange:
        ulim = 400.0
    else:
        ulim = ulim_d[crystal]
    omega_l = np.linspace(0.0,ulim,500)/Eh_to_eV

    if bigrange:
        addn = '_bigrange'
    else:
        addn = ''

    for fxc in fxcl:

        if fxc == 'QV' and use_multiprocesing:
            nproc = 4
        else:
            nproc = 1

        # only need to evaluate zero-frequency term once
        if fxc in dyn_only_regex:
            fxc_g0 = wrap_kernel(0.0,0.0,n_0_bar,fxc)
            if nproc == 1:
                # also only need to evaluate f_xc[n](q=0,omega) once
                fxc_g = wrap_kernel(0.0,omega_l,n_0_bar,fxc)
        else:
            if nproc == 1:
                fxc_g0 = wrap_kernel(gmod,0.0,n_0_bar,fxc)
            else:
                pool = mp.Pool(processes=nproc)
                fxc_g0_tmp = pool.starmap(wrap_kernel,product(gmod,[0.0],[n_0_bar],[fxc]))
                pool.close()
                fxc_g0 = np.zeros(Ng,dtype='complex')
                for ifxcg in range(Ng):
                    fxc_g0[ifxcg] = fxc_g0_tmp[ifxcg]

        alpha = np.zeros(omega_l.shape[0],dtype='complex')

        ofl = wdir+'alpha_omega_'+fxc+addn+'.csv'
        intwg = 1.0/3.0
        if fxc in dyn_only_regex and nproc > 1:
            pool = mp.Pool(processes=nproc)
            fxc_g_tmp = pool.starmap(wrap_kernel,product([0.0],omega_l,[n_0_bar],[fxc]))
            pool.close()
            fxc_g = np.zeros(omega_l.shape[0],dtype='complex')
            for ifxcg in range(omega_l.shape[0]):
                fxc_g[ifxcg] = fxc_g_tmp[ifxcg]
        for iom,om in enumerate(omega_l):
            if fxc in dyn_only_regex:
                fxc_diff = fxc_g[iom] - fxc_g0
            else:
                if nproc == 1:
                    fxc_g = wrap_kernel(gmod,om,n_0_bar,fxc)
                else:
                    pool = mp.Pool(processes=nproc)
                    fxc_g_tmp = pool.starmap(wrap_kernel,product(gmod,[om],[n_0_bar],[fxc]))
                    pool.close()
                    fxc_g = np.zeros(Ng,dtype='complex')
                    for ifxcg in range(Ng):
                        fxc_g[ifxcg] = fxc_g_tmp[ifxcg]
                fxc_diff = fxc_g - fxc_g0
            alpha[iom] = intwg*np.sum(g_dot_q_hat*fxc_diff*ng02)/n_avg2

        np.savetxt(ofl,np.transpose((omega_l,alpha.real,alpha.imag)),delimiter=',',header='omega (a.u.), Re alpha(w), Im alpha(w)')

    return

def plotter(fxcl,sign_conv=1):

    clist=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:olive','tab:gray']
    line_styles=['-','--','-.']

    if bigrange:
        olim = 400
    else:
        olim = ulim_d[crystal]
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
    if crystal == 'Si' and not bigrange:
        axins = inset_axes(ax[0],width='50%',height='45%',loc=3,borderpad=0)#InsetPosition(ax[0],[.05,.05,.45,.5]), loc=3)
    for ifxc,anfxc in enumerate(fxcl):
        flnm = wdir+'alpha_omega_'+anfxc+addn+'.csv'

        om[anfxc],alp_re[anfxc],alp_im[anfxc] = np.transpose(np.genfromtxt(flnm,delimiter=',',skip_header=1))
        om[anfxc]*=Eh_to_eV
        mask = om[anfxc]<=olim
        om[anfxc] = om[anfxc][mask]
        alp_re[anfxc] = alp_re[anfxc][mask]
        alp_im[anfxc] = alp_im[anfxc][mask]

        if sign_conv > 0:
            # positive sign for sign convention that f_xc(q,omega) ~ (alpha + beta*omega**2)/q**2, a la Nazaraov & Vignale
            max_bd = max([max_bd,alp_re[anfxc].max()])
            min_bd = min([min_bd,alp_im[anfxc].min()])
        elif sign_conv < 0:
            # negative sign for sign convention that f_xc(q,omega) ~ -(alpha + beta*omega**2)/q**2, a la Botti & Reining
            alp_re[anfxc]*= -1
            alp_im[anfxc]*= -1
            max_bd = max([max_bd,alp_im[anfxc].max()])
            min_bd = min([min_bd,alp_re[anfxc].min()])
        ax[0].plot(om[anfxc],alp_re[anfxc],color=clist[ifxc],linestyle=line_styles[ifxc%len(line_styles)])
        ax[1].plot(om[anfxc],alp_im[anfxc],color=clist[ifxc],linestyle=line_styles[ifxc%len(line_styles)])
        if crystal == 'Si' and not bigrange:
            axins.plot(om[anfxc][om[anfxc]<=10.0],alp_re[anfxc][om[anfxc]<=10.0],color=clist[ifxc],linestyle=line_styles[ifxc%len(line_styles)])
    if crystal == 'Si' and not bigrange:
        axins.set_xlim([0.0,10.0])
        axins.set_xticks([2.0,4.0,6.0,8.0])
        axins.tick_params(axis='x',direction='in',top=True,bottom=False,labelbottom=False, labeltop=True,pad=1)
        axins.tick_params(axis='y',direction='in',right=True,left=False,labelleft=False, labelright=True,pad=1)
    if 'QV' in fxcl:
        if sign_conv > 0:
            ax[0].set_ylim([1.5*alp_re['QV'].min(),1.05*max_bd])
            ax[1].set_ylim([1.05*min_bd,0.0])
        elif sign_conv < 0:
            ax[1].set_ylim([0.0,1.05*max_bd])
            ax[0].set_ylim([1.05*min_bd,1.5*alp_re['QV'].max()])
    else:
        if sign_conv > 0:
            ax[0].set_ylim([0.0,1.05*max_bd])
            ax[1].set_ylim([1.05*min_bd,0.0])
        elif sign_conv < 0:
            ax[1].set_ylim([0.0,1.05*max_bd])
            ax[0].set_ylim([1.05*min_bd,0.0])
    ax[1].set_xlabel('$\\omega$ (eV)',fontsize=16)
    ax[0].set_ylabel('$\\mathrm{Re}~\\alpha(\\omega)$',fontsize=16)
    ax[1].set_ylabel('$\\mathrm{Im}~\\alpha(\\omega)$',fontsize=16)
    axpars = {'y': {'Si': (.1,.2,.1,.2),'C':(.025,.05,.025,.05), 'Al': (.05,.1,.05,.1), 'Na': (.025,.05,.025,.05)},
    'x': {'Si': (1,2), 'C': (1,2), 'Al': (25,50), 'Na': (10,20)}}
    if crystal == 'Si':
        if bigrange:
            axpars['y']['Si'] = (.125,.25,.1,.2)
            axpars['x']['Si'] = (25,50)
    ax[0].yaxis.set_minor_locator(MultipleLocator(axpars['y'][crystal][0]))
    ax[0].yaxis.set_major_locator(MultipleLocator(axpars['y'][crystal][1]))
    ax[1].yaxis.set_minor_locator(MultipleLocator(axpars['y'][crystal][2]))
    ax[1].yaxis.set_major_locator(MultipleLocator(axpars['y'][crystal][3]))
    for i in range(2):
        ax[i].set_xlim([0.0,olim])#om.max()])
        ax[i].tick_params(axis='both',labelsize=14)
        ax[i].xaxis.set_minor_locator(MultipleLocator(axpars['x'][crystal][0]))
        ax[i].xaxis.set_major_locator(MultipleLocator(axpars['x'][crystal][1]))

    ax[0].xaxis.set_ticklabels([' ' for i in ax[0].xaxis.get_major_ticks()])
    ax[0].tick_params(axis='y',labelsize=14)
    if crystal in semicond_regex:
        plt.suptitle('{:}, r$^2$SCAN density'.format(crystal),fontsize=16)
    elif crystal in metal_regex:
        plt.suptitle('{:}, pseudopotential density'.format(crystal),fontsize=16)
    for ifxc,anfxc in enumerate(fxcl):
        offset_d = {'Si':.01,'C':.01,'Al':.01,'Na':.005}
        offset=offset_d[crystal]
        if anfxc == 'DLDA':
            lbl = 'Dynamic LDA'
            if bigrange and sign_conv > 0:
                offset = -0.35
        elif anfxc == 'MCP07_k0':
            lbl = 'MCP07, $\\overline{k}=0$'
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
        if sign_conv<0:
            fac = {'Si': {'DLDA':10, 'MCP07': -.2, 'MCP07_k0': -2.5, 'QV': 2},#{'DLDA':22, 'MCP07': 3.5, 'MCP07_k0': 14, 'QV': 2.5},
            'C': {'DLDA': 3, 'MCP07': -.1, 'MCP07_k0': .02, 'QV': 1}}
            if bigrange:
                fac = {'Si': {'DLDA':12, 'MCP07': -.2, 'MCP07_k0': -2.2, 'QV': 3}}
            if crystal in fac:
                offset *= -fac[crystal][anfxc]
        ax[0].annotate(lbl,(om[anfxc][wind],alp_re[anfxc][wind]+offset),color=clist[ifxc],fontsize=14,rotation=angle,rotation_mode='anchor')
    plt.subplots_adjust(top=.93)
    #plt.show()
    #exit()
    plt.savefig('./figs/{:}_alpha_omega'.format(crystal)+addn+'.pdf',dpi=600,bbox_inches='tight')
    return

if __name__=="__main__":
    #plot_fourier_components()
    #exit()

    #calc_alpha(['DLDA','MCP07_k0','MCP07','QV'])
    plotter(['MCP07','MCP07_k0','DLDA','QV'],sign_conv=-1)
