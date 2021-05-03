import numpy as np
from os.path import isfile
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import multiprocessing

from constants import pi
from mcp07 import exact_constraints,mcp07_dynamic,mcp07_static,gki_dynamic_real_freq,alda
from gauss_quad import gauss_quad
from qian_vignale_fxc import fxc_longitudinal_fixed_grid,density_variables,get_qv_pars

to_do_list = ['RPA','ALDA','DLDA','QV','MCP07 static','MCP07_k0','MCP07']#'QV_MCP07',

q_indep = ['DLDA','QV']
w_indep = ['MCP07 static']
freq_indep = ['RPA','ALDA']

def eps_c_pw92_unpol(rs):

    # J. P. Perdew and Y. Wang, PRB 45, 13244 (1992).
    # doi: 10.1103/PhysRevB.45.13244
    def g(v,rs):
        q0 = -2.0*v[0]*(1.0 + v[1]*rs)
        q1 = 2.0*v[0]*(v[2]*rs**(0.5) + v[3]*rs + v[4]*rs**(1.5) + v[5]*rs**2)
        return q0*np.log(1.0 + 1.0/q1)

    return g([0.031091,0.21370,7.5957,3.5876,1.6382,0.49294],rs)

def fxc_selector(q,w,dv,wfxc,grid=[],wg=[]):

    if wfxc == 'MCP07':
        fxc = mcp07_dynamic(q,w,dv,axis='real',revised=False,param='PZ81',no_k=False)
    elif wfxc == 'MCP07_k0':
        fxc = mcp07_dynamic(q,w,dv,axis='real',revised=False,param='PZ81',no_k=True)
    elif wfxc == 'MCP07 static':
        fxc,_,_ = mcp07_static(q,dv,param='PZ81')
    elif wfxc == 'DLDA':
        fxc = gki_dynamic_real_freq(dv,w,x_only=False,revised=False,param='PZ81',dimensionless=False)
    elif wfxc == 'QV':
        fxc = fxc_longitudinal_fixed_grid(w,dv,grid,wg)
    elif wfxc == 'ALDA':
        fxc = alda(dv,param='PZ81')
    elif wfxc == 'RPA':
        fxc = 0.0
    else:
        raise ValueError('Unknown XC kernel, ', wfxc)
    return fxc

def chiks_imag_freq(q,u,dv):
    #return chi_parser(q/2,w,0.0,dv['rs'],'chi0',reduce_omega=False,imag_freq=True)
    """
        Eqs. 28 and 29 of
        M. Lein, E.K.U. Gross, and J.P. Perdew,
        Phys. Rev. B 61, 13431 (2000).
        https://doi.org/10.1103/PhysRevB.61.13431
    """
    qq = q/(2*dv['kF'])
    ut = u/(q*dv['kF'])
    ut2 = ut**2
    chi0 = (qq**2 - ut2 - 1)/(4*qq)*np.log((ut2 + (qq+1)**2)/(ut2 + (qq-1)**2))
    chi0 += -1 + ut*np.arctan((1 + qq)/ut) + ut*np.arctan((1 - qq)/ut)
    return dv['kF']/(2*pi**2)*chi0

def make_grid(def_pts=200,cut_pt=2.0):

    gfile = './grids/gauss_legendre_{:}_pts.csv'
    if isfile(gfile):
        wg,grid = np.transpose(np.genfromtxt(gfile,delimiter=',',skip_header=1))
    else:
        wg,grid = gauss_quad(def_pts,grid_type='legendre')

    # first step, shift grid and weights from (-1, 1) to (0, 1)
    sgrid = 0.5*(grid + 1)
    swg = 0.5*wg

    """
        ogrid integrates from 0 to cut_pt, then extrapolates from cut_pt to infinity using
        int_0^inf f(x) dx = int_0^x_c f(x) dx + int_0^(1/x_c) f(1/u)/u**2 du,
        with u = 1/x. Then x_c = cut_pt, which by default = 2.
    """
    ogrid = cut_pt*sgrid
    owg = cut_pt*swg
    oextrap = sgrid/cut_pt
    owg_extrap = swg/(cut_pt*oextrap**2)
    ogrid = np.concatenate((ogrid,1/oextrap))
    owg = np.concatenate((owg,owg_extrap))

    return ogrid,owg,sgrid,swg

def fxc_ifreq_fixed_grid(q,omega,dv,inf_grid,inf_wg,wfxc):

    dinf_grid = np.concatenate((inf_grid,-inf_grid))
    dinf_wg = np.concatenate((inf_wg,inf_wg))

    if wfxc in ['DLDA','MCP07_k0','MCP07']:
        _,finf = exact_constraints(dv,x_only=False,param='PZ81')
    elif wfxc == 'QV':
        _,finf = exact_constraints(dv,x_only=False,param='PW92')

    if hasattr(omega,'__len__'):

        fxc_iu = np.zeros(omega.shape,dtype='complex')

        fxc_tmp = fxc_selector(q,dinf_grid,dv,wfxc,grid=inf_grid,wg=inf_wg)

        for iw,w in enumerate(omega):
            rintd = (w*(fxc_tmp.real-finf) + dinf_grid*fxc_tmp.imag)/(dinf_grid**2 + w**2)
            iintd = (-dinf_grid*(fxc_tmp.real-finf) + w*fxc_tmp.imag)/(dinf_grid**2 + w**2)
            fxc_iu[iw] = np.sum(dinf_wg*(rintd + 1.j*iintd))
            #fxc_iu.real[iw] = np.sum(dinf_wg*rintd)
            #fxc_iu.imag[iw] = np.sum(dinf_wg*iintd)
    else:

        fxc_tmp = fxc_selector(q,dinf_grid,dv,wfxc,grid=inf_grid,wg=inf_wg)
        rintd = (omega*(fxc_tmp.real-finf) + dinf_grid*fxc_tmp.imag)/(dinf_grid**2 + omega**2)
        iintd = (-dinf_grid*(fxc_tmp.real-finf) + omega*fxc_tmp.imag)/(dinf_grid**2 + omega**2)
        fxc_iu = np.sum(dinf_wg*(rintd + 1.j*iintd))

    return finf + fxc_iu/(2*pi)

def get_eps_c_fixed_grid(rs):

    dv = density_variables(rs)

    ginf,wginf,lgrid,lwg = make_grid(def_pts=200,cut_pt=50*dv['wp0'])
    qgr,qwg,lgrid,lwg = make_grid(def_pts=100,cut_pt=20*dv['kF'])
    #wgr,wwg,sgrid,swg = make_grid(def_pts=20,cut_pt=20*dv['wp0'])
    #qgr = 10*dv['kF']*lgrid
    #qwg = 10*dv['kF']*lwg
    wgr = 50*dv['wp0']*lgrid
    wwg = 50*dv['wp0']*lwg
    wex = -np.log(lgrid*np.exp(-50*dv['wp0']))
    wgr = np.concatenate((wgr,wex))
    wwg = np.concatenate((wwg,lwg/lgrid))

    ec = {}
    fxc_q_indep = {}
    fxc_w_indep = {}

    for fnl in to_do_list:
        ec[fnl] = 0.0
        if fnl in q_indep:
            fxc_q_indep[fnl] = np.zeros((lgrid.shape[0],wgr.shape[0]),dtype='complex')
            for ilam,alam in enumerate(lgrid):
                wscl = wgr/alam
                fxc_q_indep[fnl][ilam] = fxc_ifreq_fixed_grid(0.0,wscl,density_variables(rs*alam),ginf,wginf,fnl)
        elif fnl in w_indep:
            fxc_w_indep[fnl] = np.zeros((qgr.shape[0],lgrid.shape[0]),dtype='complex')
            for iq,aq in enumerate(qgr):
                qscl = aq/lgrid
                fxc_w_indep[fnl][iq] = fxc_selector(qscl,0.0,density_variables(rs*lgrid),fnl)

    if 'DLDA' not in to_do_list:
        if 'MCP07' in to_do_list or 'MCP07_k0' in to_do_list:
            fxc_q_indep['DLDA'] = np.zeros((lgrid.shape[0],wgr.shape[0]),dtype='complex')
            for ilam,alam in enumerate(lgrid):
                wscl = wgr/alam
                fxc_q_indep['DLDA'][ilam] = fxc_ifreq_fixed_grid(0.0,wscl,density_variables(rs*alam),ginf,wginf,'DLDA')
    if 'QV' not in to_do_list and 'QV_MCP07' in to_do_list:
        fxc_q_indep['QV'] = np.zeros((lgrid.shape[0],wgr.shape[0]),dtype='complex')
        for ilam,alam in enumerate(lgrid):
            wscl = wgr/alam
            fxc_q_indep['QV'][ilam] = fxc_ifreq_fixed_grid(0.0,wscl,density_variables(rs*alam),ginf,wginf,'QV')

    chi0m = np.zeros((qgr.shape[0],wgr.shape[0]))
    for iq,aq in enumerate(qgr):
        chi0m[iq] = chiks_imag_freq(aq,wgr,dv)

    for iq,aq in enumerate(qgr):
        for ilam,alam in enumerate(lgrid):

            int_wg = qwg[iq]*lwg[ilam]*wwg/(pi**2*dv['n'])

            q_scl = aq/alam
            w_scl = wgr/alam**2
            dv_scl = density_variables(rs*alam)
            vc_scl = 4*pi*alam/aq**2

            for fnl in to_do_list:
                if fnl in freq_indep:
                    fxc = fxc_selector(q_scl,w_scl,dv_scl,fnl)
                elif fnl in q_indep:
                    fxc = fxc_q_indep[fnl][ilam]
                elif fnl in w_indep:
                    fxc = fxc_w_indep[fnl][iq,ilam]
                elif fnl == 'MCP07':
                    fxc_q,f0,akn = mcp07_static(q_scl,dv_scl,param='PZ81')
                    fxc = (1.0 + np.exp(-akn*q_scl**2)*(fxc_q_indep['DLDA'][ilam]/f0 - 1.0))*fxc_q
                elif fnl == 'MCP07_k0':
                    fxc_q,f0,akn = mcp07_static(q_scl,dv_scl,param='PZ81')
                    fxc = fxc_q_indep['DLDA'][ilam]/f0*fxc_q
                elif fnl == 'QV_MCP07':
                    fxc_q,f0_gki,akn = mcp07_static(q_scl,dv_scl,param='PW92')
                    f0_qv = fxc_selector(0.0,0.0,dv_scl,'QV')
                    fxc = (f0_qv + np.exp(-akn*q_scl**2)*(fxc_q_indep['QV'][ilam] - f0_qv))*fxc_q/f0_gki
                else:
                    fxc = fxc_ifreq_fixed_grid(q_scl,w_scl,dv_scl,ginf,wginf,fnl)
                fxch = fxc/alam + vc_scl
                # Eq. 27 of Lein, Gross, and Perdew
                ec[fnl] -= np.sum(chi0m[iq]**2*fxch/(1 - chi0m[iq]*fxch)*int_wg)

    for fnl in to_do_list:
        if abs(ec[fnl].imag) < 1.e-15:
            ec[fnl] = ec[fnl].real

    ec['PW92'] = eps_c_pw92_unpol(rs)
    return ec

def eps_c_plots():

    #clist=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:olive','tab:gray']
    ['darkblue','darkorange','darkgreen','darkred','black']
    clist = {'PW92':'black', 'RPA': 'darkblue', 'ALDAxc': 'purple', 'DLDA': 'tab:green', 'QV': 'tab:red',
    'QV_MCP07': 'brown','MCP07 static': 'tab:green', 'MCP07_k0': 'tab:orange', 'MCP07': 'tab:blue',}
    line_styles={'PW92':'-', 'RPA': '--', 'ALDAxc': ':', 'DLDA': '-.', 'QV': '-', 'QV_MCP07': '-.',
    'MCP07 static': ':', 'MCP07_k0': '--', 'MCP07': '-'}#['-','--','-.',':']
    mkrlist=['o','s','d','^','v','x','*','+']

    fig,ax = plt.subplots(figsize=(8,6))
    epsc = {}
    rs,epsc['PW92'],epsc['RPA'],epsc['ALDAxc'],epsc['DLDA'],epsc['QV'],_,epsc['QV_MCP07'],_,_,epsc['MCP07_k0'],epsc['MCP07'] = np.transpose(np.genfromtxt('./data_files/jellium_eps_c.csv',delimiter=',',skip_header=1))

    ax.set_xlim([rs.min(),rs.max()])
    ax.set_ylim([-0.08,0.0])
    for ifnl,fnl in enumerate(epsc):
        epsc[fnl] = epsc[fnl].real
        lbl = fnl
        if fnl == 'DLDA':
            lbl = 'Dynamic LDA'
        elif fnl == 'MCP07_k0':
            lbl = 'MCP07, $\\overline{k}=0$'
        ax.plot(rs,epsc[fnl],markersize=5,color=clist[fnl],linestyle=line_styles[fnl],label=lbl,linewidth=2.5)
        if fnl in ['ALDAxc','RPA','DLDA']:
            i = (rs.shape[0]-rs.shape[0]%2)//2
            offset = .0005
            p1 = ax.transData.transform_point((rs[i],epsc[fnl][i]))
            p2 = ax.transData.transform_point((rs[i+1],epsc[fnl][i+1]))
            angle = 180/pi*np.arctan((p2[1]-p1[1])/(p2[0]-p1[0]))
            ax.annotate(lbl,(0.5*(rs[i]+rs[i+1])-len(fnl)*0.1,0.5*(epsc[fnl][i]+epsc[fnl][i+1])+offset),rotation=angle,color=clist[fnl],fontsize=16)
        elif fnl in ['MCP07','MCP07_k0']:
            if fnl == 'MCP07':
                i = (rs.shape[0]-rs.shape[0]%4)//4
                txtpos = (2,-0.07)
            else:
                i = (rs.shape[0]-rs.shape[0]%3)//3
                txtpos = (3.6,-0.06)
            ax.annotate(lbl,(0.5*(rs[i]+rs[i+1]),0.5*(epsc[fnl][i]+epsc[fnl][i+1])),xytext=txtpos,color=clist[fnl],fontsize=16,arrowprops=dict(linewidth=2,color=clist[fnl],arrowstyle='->'))
        elif fnl == 'QV':
            ax.annotate(lbl,(2.02,-0.037),color=clist[fnl],fontsize=16)
        elif fnl == 'PW92':
            i = 3*(rs.shape[0]-rs.shape[0]%4)//4
            ax.annotate(lbl,(rs[i],epsc['PW92'][i]),(8,-0.04),color=clist[fnl],fontsize=16,arrowprops=dict(linewidth=2,color=clist[fnl],arrowstyle='->'))
    #ax.legend(fontsize=18)
    ax.yaxis.set_major_locator(MultipleLocator(.01))
    ax.yaxis.set_minor_locator(MultipleLocator(.005))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel('$r_{\\mathrm{s}}$ (bohr)',fontsize=24)
    ax.set_ylabel('$\\varepsilon_{\\mathrm{c}}$ (hartree)',fontsize=24)
    ax.tick_params(axis='both',labelsize=20)
    #plt.show()
    plt.savefig('./figs/ueg_epsilon_c.pdf',dpi=600,bbox_inches='tight')
    plt.savefig('./eps_figs/ueg_epsilon_c.eps',dpi=600,bbox_inches='tight')
    return


if __name__=="__main__":

    #eps_c_plots()
    #exit()

    nproc = 4
    rss = np.linspace(1,10,101)

    if nproc > 1:
        pool = multiprocessing.Pool(processes=nproc)
        ecdd = pool.map(get_eps_c_fixed_grid,rss)
        pool.close()

    ofl = open('./data_files/jellium_eps_c.csv','w+')
    str = 'rs, PW92, '
    for ifnl,fnl in enumerate(to_do_list):
        if fnl == 'QV' or fnl == 'QV_MCP07':
            if ifnl < len(to_do_list)-1:
                str += 'Re {:}, Im {:},'.format(fnl,fnl)
            else:
                str += 'Re {:}, Im {:} \n'.format(fnl,fnl)
        else:
            if ifnl < len(to_do_list)-1:
                str += '{:}, '.format(fnl)
            else:
                str += '{:} \n'.format(fnl)
    ofl.write(str)

    for irs,rs in enumerate(rss):
        if nproc == 1:
            ecd = get_eps_c_fixed_grid(rs)
        else:
            ecd = ecdd[irs]
        str = '{:}, {:}, '.format(rs,ecd['PW92'])
        for ifnl,fnl in enumerate(to_do_list):
            if fnl == 'QV' or fnl == 'QV_MCP07':
                if ifnl < len(to_do_list)-1:
                    str += '{:}, {:}, '.format(ecd[fnl].real,ecd[fnl].imag)
                else:
                    str += '{:}, {:} \n'.format(ecd[fnl].real,ecd[fnl].imag)
            else:
                if ifnl < len(to_do_list)-1:
                    str += '{:}, '.format(ecd[fnl])
                else:
                    str += '{:} \n'.format(ecd[fnl])
        ofl.write(str)
    ofl.close()

    """
    rs = 3
    dv = density_variables(rs)
    ginf,wginf,lgrid,lwg = make_grid(def_pts=200,cut_pt=50*dv['wp0'])
    dinf = np.concatenate((ginf,-ginf))
    dinfwg = np.concatenate((wginf,wginf))


    freqs = np.linspace(0,50*dv['wp0'],2000)
    fxc = fxc_longitudinal_fixed_grid(freqs,dv,ginf,wginf)#fxc_ifreq_fixed_grid(0.0,freqs,density_variables(4),dinf,dinfwg,'QV')

    import matplotlib.pyplot as plt
    plt.plot(freqs/dv['wp0'],fxc.real/(2*dv['wp0']/dv['n']))
    plt.plot(freqs/dv['wp0'],fxc.imag/(2*dv['wp0']/dv['n']))
    plt.show()
    exit()
    """
