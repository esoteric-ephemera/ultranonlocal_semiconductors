import numpy as np
from os import path,mkdir
from itertools import product
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from constants import crystal,pi,Eh_to_eV,clist
from mcp07 import chi_parser,mcp07_dynamic,mcp07_static,gki_dynamic_real_freq
from qian_vignale_fxc import fxc_longitudinal as qv_fxc
from pseudo_pt import get_n_g
from fhnc_2p2h import fxc_2p2h_lin_interp

metal_regex = ['Al','Na']
semicond_regex = ['Si','C']
if crystal in semicond_regex:
    from discrete_ft_n import dft_n
    from discrete_ft_n import oflnm as ft_dens_file

bigrange = False
use_multiprocesing = True
for adir in ['./figs', './eps_figs']:
    if not path.isdir(adir):
        mkdir(adir)
wdir = './data_files/{:}/'.format(crystal)

ulim_d = {'Si':25.0, 'C':25.0, 'Al': 250.0, 'Na': 100.0}

def get_len(vec):
    return np.sum(vec**2)**(0.5)

def get_dens_vars_from_n(n):
    dv = {}
    dv['n'] = n
    dv['kF'] = (3*pi**2*n)**(1/3)
    dv['rs'] = (3/(4*pi*n))**(1/3)
    dv['rsh'] = dv['rs']**(0.5)
    dv['wp0'] = (3/dv['rs']**3)**(0.5)
    return dv

def wrap_kernel(q,omega,n,wfxc):
    dv = get_dens_vars_from_n(n)
    if wfxc == 'MCP07':
        fxc = mcp07_dynamic(q,omega,dv,axis='real',param='PZ81',no_k=False)
    elif wfxc == 'MCP07_k0':
        fxc = mcp07_dynamic(q,omega,dv,axis='real',param='PZ81',no_k=True)
    elif wfxc == 'DLDA':
        fxc=gki_dynamic_real_freq(dv,omega,x_only=False,param='PZ81',dimensionless=False)
    elif wfxc == 'QV':
        fxc = qv_fxc(dv,omega,use_mu_xc=True)
    elif wfxc == 'QV_MCP07_TD' or wfxc == 'QV_TD':
        fxc = qv_fxc(dv,omega,use_mu_xc=False)
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


def calc_alpha(fxcl):

    dyn_only_regex = ['DLDA','QV']

    gmod,ng02,n_0_bar = init_n_g(crystal)
    print(('average density {:} bohr**(-3); rs_avg = {:} bohr').format(n_0_bar,(3.0/(4*pi*n_0_bar))**(1/3)))
    n_avg2 = n_0_bar**2
    gmod2 = gmod**2

    Ng = gmod.shape[0]
    if bigrange:
        ulim = 400.0
    else:
        ulim = ulim_d[crystal]
    Nw = 500
    omega_l = np.linspace(0.0,ulim,Nw)/Eh_to_eV

    if bigrange:
        addn = '_bigrange'
    else:
        addn = ''

    if use_multiprocesing:
        nproc = 4
    else:
        nproc = 1

    if '2p2h' in fxcl:
        wcut_2p2h = 3.98*(4*pi*n_0_bar)**(0.5)
        fxcdyn = fxc_2p2h_lin_interp((3/(4*pi*n_0_bar))**(1/3),gmod,omega_l)

    fxc_g0_dyn = {}
    fxc_g_dyn = {}
    tmp_multi = []
    if 'QV_MCP07_TD' in fxcl:
        tmp_multi.append('QV_MCP07_TD')
    if 'QV' in fxcl:
        tmp_multi.append('QV')
    for fxc in tmp_multi:
        pool = mp.Pool(processes=nproc)
        fxc_g_tmp = pool.starmap(wrap_kernel,product([0.0],omega_l,[n_0_bar],[fxc]))
        pool.close()
        fxc_g_dyn[fxc] = np.zeros(Nw,dtype='complex')
        for ifxc in range(Nw):
            fxc_g_dyn[fxc][ifxc] = fxc_g_tmp[ifxc]

        if omega_l[0]==0.0:
            fxc_g0_dyn[fxc] = fxc_g_dyn[fxc][0]
        else:
            fxc_g0_dyn[fxc] = wrap_kernel(0.0,0.0,n_0_bar,fxc)

    if 'DLDA' in fxcl:
        fxc_g0_dyn['DLDA'] = wrap_kernel(0.0,0.0,n_0_bar,'DLDA')
        fxc_g_dyn['DLDA'] = wrap_kernel(0.0,omega_l,n_0_bar,'DLDA')

    for fxc in fxcl:

        if fxc in dyn_only_regex:
            fxc_g0 = fxc_g0_dyn[fxc]
            fxc_g = fxc_g_dyn[fxc]
        elif fxc == 'QV_MCP07_TD':
            fxc_stat_tmp,f_alda,k_mcp07 = mcp07_static(gmod,get_dens_vars_from_n(n_0_bar),param='PW92')
        elif fxc not in ['2p2h']:
            fxc_g0 = wrap_kernel(gmod,0.0,n_0_bar,fxc)

        alpha = np.zeros(Nw,dtype='complex')

        ofl = wdir+'alpha_omega_'+fxc+addn+'.csv'

        if fxc in dyn_only_regex:
            fxc_diff = fxc_g - fxc_g0
            rlv_sum = np.sum(gmod2*ng02)/(3*n_avg2)
            alpha = rlv_sum*fxc_diff
        else:

            for iom,om in enumerate(omega_l):

                if fxc == 'QV_MCP07_TD':
                    fxc_g0 = (1 + np.exp(-k_mcp07*gmod**2)*(fxc_g0_dyn['QV_MCP07_TD']/f_alda - 1))*fxc_stat_tmp
                    fxc_g = (1 + np.exp(-k_mcp07*gmod**2)*(fxc_g_dyn['QV_MCP07_TD'][iom]/f_alda - 1))*fxc_stat_tmp
                    fxc_diff = fxc_g - fxc_g0
                elif fxc == '2p2h':
                    # maximum tabulated value for 2p2h kernel is 3.98 omega_p(0)
                    if om <= wcut_2p2h:
                        fxc_diff = fxcdyn[:,iom] - fxcdyn[:,0]
                    else:
                        break
                else:
                    fxc_g = wrap_kernel(gmod,om,n_0_bar,fxc)
                    fxc_diff = fxc_g - fxc_g0
                alpha[iom] = np.sum(gmod2*fxc_diff*ng02)/(3*n_avg2)

        if fxc == '2p2h':
            wmask = omega_l <= wcut_2p2h
            np.savetxt(ofl,np.transpose((omega_l[wmask],alpha.real[wmask],alpha.imag[wmask])),delimiter=',',header='omega (a.u.), Re alpha(w), Im alpha(w)')
        else:
            np.savetxt(ofl,np.transpose((omega_l,alpha.real,alpha.imag)),delimiter=',',header='omega (a.u.), Re alpha(w), Im alpha(w)')


    return

def plotter(fxcl,sign_conv=1):

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
        alp_re[anfxc] = sign_conv*alp_re[anfxc][mask]
        alp_im[anfxc] = sign_conv*alp_im[anfxc][mask]

        if sign_conv > 0:
            # positive sign for sign convention that f_xc(q,omega) ~ (alpha + beta*omega**2)/q**2, a la Nazaraov & Vignale
            max_bd = max([max_bd,alp_re[anfxc].max()])
            min_bd = min([min_bd,alp_im[anfxc].min()])
        elif sign_conv < 0:
            # negative sign for sign convention that f_xc(q,omega) ~ -(alpha + beta*omega**2)/q**2, a la Botti & Reining
            max_bd = max([max_bd,alp_im[anfxc].max()])
            min_bd = min([min_bd,alp_re[anfxc].min()])
        ax[0].plot(om[anfxc],alp_re[anfxc],color=clist[ifxc],linestyle=line_styles[ifxc%len(line_styles)],linewidth=2,label=anfxc)
        ax[1].plot(om[anfxc],alp_im[anfxc],color=clist[ifxc],linestyle=line_styles[ifxc%len(line_styles)],linewidth=2,label=anfxc)

        if crystal == 'Si' and not bigrange:
            axins.plot(om[anfxc][om[anfxc]<=10.0],alp_re[anfxc][om[anfxc]<=10.0],color=clist[ifxc],linestyle=line_styles[ifxc%len(line_styles)],linewidth=1.5)
    if crystal == 'Si' and not bigrange:
        axins.set_xlim([0.0,10.0])
        axins.set_xticks([2.0,4.0,6.0,8.0])
        axins.tick_params(axis='x',direction='in',top=True,bottom=False,labelbottom=False, labeltop=True,pad=1)
        axins.tick_params(axis='y',direction='in',right=True,left=False,labelleft=False, labelright=True,pad=1)
        axins.tick_params(axis='both',labelsize=16)
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
    ax[1].set_xlabel('$\\omega$ (eV)',fontsize=20)
    ax[0].set_ylabel('$\\mathrm{Re}~\\alpha(\\omega)$',fontsize=20)
    ax[1].set_ylabel('$\\mathrm{Im}~\\alpha(\\omega)$',fontsize=20)
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
        ax[i].tick_params(axis='both',labelsize=20)
        ax[i].xaxis.set_minor_locator(MultipleLocator(axpars['x'][crystal][0]))
        ax[i].xaxis.set_major_locator(MultipleLocator(axpars['x'][crystal][1]))

    if crystal in semicond_regex:

        # alpha is dimensionless
        # beta given in unites of 10**(-3) * eV**(-2)
        lrc_pars = {'Si': [0.13,0.00635], 'C': [0.28,0.00135]}
        apar = lrc_pars[crystal][0]
        bpar = lrc_pars[crystal][1]#*Eh_to_eV**2

        #ax[0].plot(om[anfxc],apar + bpar*om[anfxc]**2,color=clist[ifxc+1],linestyle=line_styles[ifxc%len(line_styles)],linewidth=2,label='LRC')
        ax[0].hlines(apar,ax[0].get_xlim()[0],ax[0].get_xlim()[1],linestyle='--',color='gray',linewidth=1)
        if crystal == 'Si' and not bigrange:
            axins.hlines(apar,axins.get_xlim()[0],axins.get_xlim()[1],linestyle='--',color='gray',linewidth=1)

    ax[0].xaxis.set_ticklabels([' ' for i in ax[0].xaxis.get_major_ticks()])
    ax[0].tick_params(axis='y',labelsize=20)
    if crystal in semicond_regex:
        plt.suptitle('{:}, r$^2$SCAN density'.format(crystal),fontsize=16)
    elif crystal in metal_regex:
        plt.suptitle('{:}, pseudopotential density'.format(crystal),fontsize=16)
    for ifxc,anfxc in enumerate(fxcl):
        scl = 0.6
        if bigrange and anfxc=='QV_MCP07_TD':
            scl = 0.3
        if crystal == 'C' and anfxc == '2p2h':
            scl = 0.85
        wind = np.argmin(np.abs(om[anfxc] - scl*olim))
        if wind > om[anfxc].shape[0]-2:
            wind = om[anfxc].shape[0]-2
        if anfxc in ['QV','QV_MCP07_TD','2p2h'] and not bigrange and crystal in semicond_regex:
            iax = 1
            tp1 = alp_im[anfxc][wind-1]
            tp2 = alp_im[anfxc][wind+1]
        else:
            iax = 0
            tp1 = alp_re[anfxc][wind-1]
            tp2 = alp_re[anfxc][wind+1]
        p2 = ax[iax].transData.transform_point((om[anfxc][wind+1],tp2))
        p1 = ax[iax].transData.transform_point((om[anfxc][wind-1],tp1))

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
        elif anfxc == 'QV_MCP07_TD':
            lbl = 'QV-MCP07-TD'
            offset = -0.04
        else:
            lbl = anfxc
            if crystal == 'Si':
                offset = 0.04
            elif crystal == 'C':
                offset = 0.02

        angle = 180/pi*np.arctan((p2[1]-p1[1])/(p2[0]-p1[0]))
        if sign_conv<0:
            fac = {'Si': {'DLDA':10, 'MCP07': -.2, 'MCP07_k0': -2.5, 'QV': -6.2, 'QV_MCP07_TD': 3.,'2p2h':-.7},#{'DLDA':22, 'MCP07': 3.5, 'MCP07_k0': 14, 'QV': 2.5},
            'C': {'DLDA': 3, 'MCP07': -.1, 'MCP07_k0': .02, 'QV': -.7, 'QV_MCP07_TD': -.5,'2p2h':-.5}}
            if bigrange:
                fac = {'Si': {'DLDA':12, 'MCP07': -.5, 'MCP07_k0': -2.2, 'QV': 3, 'QV_MCP07_TD': 1.5,'2p2h':-1}}
            if crystal in fac:
                offset *= -fac[crystal][anfxc]
        ax[iax].annotate(lbl,(om[anfxc][wind],alp_re[anfxc][wind]+offset),color=clist[ifxc],fontsize=14,rotation=angle,rotation_mode='anchor')
    #ax[1].legend(fontsize=18,ncol=3,bbox_to_anchor=(0.5, -.4),loc='center')
    ax[0].hlines(0,0.0,olim,color='gray',linewidth=1)
    plt.subplots_adjust(top=.93)
    #plt.show()
    #exit()
    plt.savefig('./figs/{:}_alpha_omega'.format(crystal)+addn+'.pdf',dpi=600,bbox_inches='tight')
    # easier to look at PDF's, but journals require eps figures
    plt.savefig('./eps_figs/{:}_alpha_omega'.format(crystal)+addn+'.eps',dpi=600,bbox_inches='tight')
    return

if __name__=="__main__":

    #plot_fourier_components()
    #exit()
    fnl_l = ['MCP07','MCP07_k0','DLDA','QV','QV_MCP07_TD','2p2h']
    calc_alpha(['QV_MCP07_TD'])
    #plotter(fnl_l,sign_conv=-1)
