import numpy as np

import constants
from integrators import nquad

pi = constants.pi
# universal constants in this module

gam = 1.311028777146059809410871821455657482147216796875
# NB: got this value from julia using the following script
# using SpecialFunctions
# BigFloat((gamma(0.25))^2/(32*pi)^(0.5))
cc = 23.0*pi/15.0

# From J. P. Perdew and Alex Zunger,
# Phys. Rev. B 23, 5048, 1981
# doi: 10.1103/PhysRevB.23.5048
# for rs < 1
au = 0.0311
bu = -0.048
cu = 0.0020
du = -0.0116

# for rs > 1
gu = -0.1423
b1u = 1.0529
b2u = 0.3334
gp = -0.0843

# From J. P. Perdew and W. Yang
# Phys. Rev. B 45, 13244 (1992).
# doi: 10.1103/PhysRevB.45.13244
A = 0.0310906908696549008630505284145328914746642112731933593
# from julia BigFloat((1-log(2))/pi^2)
alpha = 0.21370
beta1 = 7.5957
beta2 = 3.5876
beta3 = 1.6382
beta4 = 0.49294

b = (3.0/(4.0*pi))**(1.0/3.0)

def chi_parser(z,omega,ixn,rs,wfxc,reduce_omega=False,imag_freq=False,ret_eps=False,pars={},LDA='PZ81'):

    dvars = {}
    q_ixn_vec = hasattr(ixn,'__len__')
    if q_ixn_vec:
        dvars['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs*np.ones(ixn.shape)
        dvars['rs'] = rs*np.ones(ixn.shape)
    else:
        dvars['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs
        dvars['rs'] = rs
    dvars['brs']= rs

    ef = dvars['kF']**2/2.0
    dvars['ef'] = ef
    q = 2*dvars['kF']*z
    if reduce_omega:
        ufreq = omega/(4*z)
    else:
        ufreq = omega/(4*z*ef)
    chi0 = -dvars['kF']/pi**2*lindhard(z,ufreq)

    if wfxc == 'chi0':
        return chi0
    if not q_ixn_vec and ixn == 0.0:
        return chi0

    vc = 4*pi*ixn/q**2

    chi = chi0
    # if interaction strength is positive, appropriately scale
    # also need to know if it is a vector or a scalar
    if q_ixn_vec:
        dvars['rs'][ixn>0.0] *= ixn[ixn>0.0]
        dvars['kF'][ixn>0.0] /= ixn[ixn>0.0]
        q[ixn>0.0] /= ixn[ixn>0.0]
        omega[ixn>0.0] /= ixn[ixn>0.0]**2
    else:
        dvars['rs'] *= ixn
        dvars['kF'] /= ixn
        q /= ixn
        omega /= ixn**2

    dvars['n'] = 3.0/(4.0*pi*dvars['rs']**3)
    dvars['rsh'] = dvars['rs']**(0.5)

    if imag_freq:
        which_axis = 'imag'
        om = omega
    else:
        which_axis = 'real'
        om = omega.real
    if reduce_omega:
        om *= ef

    if wfxc == 'ALDA':
        fxc = alda(dvars,param=LDA)
    elif wfxc == 'RPA':
        if hasattr(omega,'__len__'):
            fxc = np.zeros(omega.shape)
        else:
            fxc = 0.0
    elif wfxc == 'MCP07':
        fxc = mcp07_dynamic(q,om,dvars,axis=which_axis,param='PZ81')
    elif wfxc == 'static MCP07':
        fxc,_,_ = mcp07_static(q,dvars,param='PZ81')
    elif wfxc == 'rMCP07':
        fxc = mcp07_dynamic(q,om,dvars,axis=which_axis,revised=True,pars=pars,param=LDA)
    elif wfxc == 'GKI':
        fxc = gki_dynamic(dvars,om,axis=which_axis,revised=True,param=LDA,use_par=True)
        #fxc =  alda(dvars,param=LDA) + (np.exp(-dvars['rs'])-1.0)*fxc
    else:
        raise SystemExit('WARNING, unrecognized XC kernel',wfxc)

    if q_ixn_vec:
        fxc[ixn>0.0] /= ixn[ixn>0.0]
    else:
        fxc /= ixn

    eps = 1.0 - (vc + fxc)*chi0
    if ret_eps:
        return eps

    if q_ixn_vec:
        chi[ixn>0.0] = chi0[ixn>0.0]/eps[ixn>0.0]
    else:
        chi = chi0/eps

    return chi

def mcp07_dynamic(q,omega,dv,axis='real',revised=False,pars={},param='PZ81',no_k=False):

    fxc_q,f0,akn = mcp07_static(q,dv,param=param)
    if revised:
        if len(pars) > 0:
            fp = pars
        else:
            raise SystemExit('rMCP07 kernel requires fit parameters!')
        kscr = fp['a']*dv['kF']/(1.0 + fp['b']*dv['kF']**(0.5))
        #F1 = (fp['a'] + fp['b']*fp['c']*dv['rs'])/(1.0 + fp['c']*dv['rs'])*dv['rs']**2
        #rs_interp = (fp['a'] + fp['b']*fp['d']*dv['rs']**fp['c'])/(1.0 + fp['d']*dv['rs']**fp['c'])
        F1 = fp['c']*dv['rs']**2#3/(1.0 + fp['d']*dv['rs'])#(1.0 + fp['c']*dv['rs']**3)/(1.0 + fp['d']*dv['rs'])
        F2 = F1 + (1.0 - F1)*np.exp(-fp['d']*(q/kscr)**2)
        #ixn_inv = dv['rs']**2*q*omega**(0.5)
        #F2 = 1.0 + (fp['c'] - 1.0)*ixn_inv/(1.0 + ixn_inv)
        fxc_omega = gki_dynamic(dv,F2*omega,axis=axis,revised=revised,param=param,use_par=True)
        fxc = (1.0 + np.exp(-(q/kscr)**2)*(fxc_omega/f0 - 1.0))*fxc_q
    else:
        fxc_omega = gki_dynamic(dv,omega,axis=axis,revised=revised,param=param,use_par=True)
        if no_k:
            if hasattr(akn,'__len__'):
                akn = np.zeros(akn.shape)
            else:
                akn = 0.0
        fxc = (1.0 + np.exp(-akn*q**2)*(fxc_omega/f0 - 1.0))*fxc_q

    return fxc

def lindhard(z,uu):

    zu1 = z - uu + 0.0j
    zu2 = z + uu + 0.0j

    fx = 0.5 + 0.0j
    fx += (1.0-zu1**2)/(8.0*z)*np.log((zu1 + 1.0)/(zu1 - 1.0))
    fx += (1.0-zu2**2)/(8.0*z)*np.log((zu2 + 1.0)/(zu2 - 1.0))

    return fx

def alda(dv,x_only=False,param='PZ81'):

    n = dv['n']
    kf = dv['kF']
    rs = dv['rs']
    rsh = dv['rsh']

    fx = -pi/kf**2

    # The uniform electron gas adiabatic correlation kernel according to
    if param == 'PZ81':
        # Perdew and Zunger, Phys. Rev. B, 23, 5076 (1981)
        if x_only:
            return fx
        else:
            if hasattr(rs,'__len__'):
                fc = np.zeros(rs.shape)

            fc_lsr = -(3*au + 2*cu*rs*np.log(rs) + (2*du + cu)*rs)/(9*n)

            fc_gtr = 5*b1u*rsh + (7*b1u**2 + 8*b2u)*rs + 21*b1u*b2u*rsh**3 + (4*b2u*rs)**2
            fc_gtr *= gu/(36*n)/(1.0 + b1u*rsh + b2u*rs)**3

            if hasattr(rs,'__len__'):
                fc[rs < 1.0] = fc_lsr[rs < 1.0]
                fc[rs >= 1.0] = fc_gtr[rs >= 1.0]
            else:
                fc = fc_gtr
                if rs < 1.0:
                    fc = fc_lsr

    elif param == 'PW92':
        # J. P. Perdew and W. Yang, Phys. Rev. B 45, 13244 (1992).
        q = 2*A*(beta1*rsh + beta2*rs + beta3*rsh**3 + beta4*rs**2)
        dq = A*(beta1/rsh + 2*beta2 + 3*beta3*rsh + 4*beta4*rs)
        ddq = A*(-beta1/2.0/rsh**3 + 3.0/2.0*beta3/rsh + 4*beta4)

        d_ec_d_rs = 2*A*( -alpha*np.log(1.0 + 1.0/q) + (1.0 + alpha*rs)*dq/(q**2 + q) )
        d2_ec_d_rs2 = 2*A/(q**2 + q)*(  2*alpha*dq + (1.0 + alpha*rs)*( ddq - (2*q + 1.0)*dq**2/(q**2 + q) )  )

        fc = rs/(9.0*n)*(rs*d2_ec_d_rs2 - 2*d_ec_d_rs)

    return fx + fc

def lda_derivs(dv,param='PZ81'):
    rs = dv['rs']
    n = dv['n']
    kf = dv['kF']
    rsh = dv['rsh']

    if param == 'PZ81':
        eps_c = gu/(1.0 + b1u*rsh + b2u*rs)
        eps_c_lsr = au*np.log(rs) + bu + cu*rs*np.log(rs) + du*rs
        if hasattr(rs,'__len__'):
            eps_c[rs < 1.0] = eps_c_lsr[rs < 1.0]
        else:
            if rs < 1.0:
                eps_c = eps_c_lsr[rs < 1.0]

        d_eps_c_d_rs = -gu*(0.5*b1u/rsh + b2u)/(1.0 + b1u*rsh + b2u*rs)**2
        d_ec_drs_lsr = au/rs + cu + cu*np.log(rs) + du
        if hasattr(rs,'__len__'):
            d_eps_c_d_rs[rs < 1.0] = d_ec_drs_lsr[rs < 1.0]
        else:
            if rs < 1.0:
                d_eps_c_d_rs = d_ec_drs_lsr

    elif param == 'PW92':
        q = 2*A*(beta1*rsh + beta2*rs + beta3*rsh**3 + beta4*rs**2)
        dq = A*(beta1/rsh + 2*beta2 + 3*beta3*rsh + 4*beta4*rs)

        eps_c = -2*A*(1.0 + alpha*rs)*np.log(1.0 + 1.0/q)
        d_eps_c_d_rs = 2*A*( -alpha*np.log(1.0 + 1.0/q) + (1.0 + alpha*rs)*dq/(q**2 + q) )

    else:
        raise SystemExit('Unknown LDA, ',param)

    return eps_c,d_eps_c_d_rs


def mcp07_static(q,dv,param='PZ81'):

    rs = dv['rs']
    n = dv['n']
    kf = dv['kF']
    rsh = dv['rsh']
    cfac = 4*pi/kf**2

    # bn according to the parametrization of Eq. (7) of
    # Massimiliano Corradini, Rodolfo Del Sole, Giovanni Onida, and Maurizia Palummo
    # Phys. Rev. B 57, 14569 (1998)
    # doi: 10.1103/PhysRevB.57.14569
    bn = 1.0 + 2.15*rsh + 0.435*rsh**3
    bn /= 3.0 + 1.57*rsh + 0.409*rsh**3

    f0 = alda(dv,param=param)
    akn = -f0/(4.0*pi*bn)

    ec,d_ec_d_rs = lda_derivs(dv,param=param)
    d_rs_ec_drs = ec + rs*d_ec_d_rs
    # The rs-dependent cn, multiplicative factor of d( r_s eps_c)/d(r_s)
    # eps_c is correlation energy per electron

    cn = -pi/(2.0*kf)*d_rs_ec_drs

    # The gradient term
    cxcn = 1.0 + 3.138*rs + 0.3*rs**2
    cxcd = 1.0 + 3.0*rs + 0.5334*rs**2
    cxc = -0.00238 + 0.00423*cxcn/cxcd
    dd = 2.0*cxc/(n**(4.0/3.0)*(4.0*pi*bn)) - 0.5*akn**2

    # The MCP07 kernel
    vc = 4.0*pi/q**2
    cl = vc*bn
    zp = akn*q**2
    grad = 1.0 + dd*q**4
    cutdown = 1.0 + 1.0/(akn*q**2)**2
    fxcmcp07 = cl*(np.exp(-zp)*grad - 1.0) - cfac*cn/cutdown

    return fxcmcp07,f0,akn

def exact_constraints(dv,x_only=False,param='PZ81'):

    ctil = -3.0/(4*pi)*(3*pi**2)**(1.0/3.0)

    n = dv['n']
    kf = dv['kF']
    rs = dv['rs']

    f0 = alda(dv,x_only=x_only,param=param)
    """
     from Iwamato and Gross, Phys. Rev. B 35, 3003 (1987),
     f(q,omega=infinity) = -4/5 n^(2/3)*d/dn[ eps_xc/n^(2/3)] + 6 n^(1/3) + d/dn[ eps_xc/n^(1/3)]
     eps_xc is XC energy per electron
    """

    # exchange contribution is -1/5 [3/(pi*n^2)]^(1/3)
    finf_x = -1.0/(5.0)*(3.0/(pi*n**2))**(1.0/3.0)

    # correlation contribution is -[22*eps_c + 26*rs*(d eps_c / d rs)]/(15*n)
    if x_only:
        finf = finf_x
    else:
        eps_c,d_eps_c_d_rs = lda_derivs(dv,param=param)
        finf_c = -(22.0*eps_c + 26.0*rs*d_eps_c_d_rs)/(15.0*n)
        finf = finf_x + finf_c

    bfac = (gam/cc)**(4.0/3.0)
    deltaf = finf - f0
    if hasattr(rs,'__len__'):
        deltaf[deltaf < 1.e-14] = 1.e-14
    else:
        if deltaf < 1.e-14:
            deltaf = 1.e-14
    bn = bfac*deltaf**(4.0/3.0)

    return bn,finf

def gki_dynamic_real_freq(dv,u,x_only=False,revised=False,param='PZ81',dimensionless=False):

    if dimensionless:
        xk = u
    else:
        # The exact constraints are the low and high-frequency limits
        bn,finf = exact_constraints(dv,x_only=x_only,param=param)
        bnh = bn**(0.5)

        # for real frequency only! gki_dynamic analytically continues this to
        # purely imaginary frequency
        xk = bnh*u

    gx = xk/((1.0 + xk**2)**(5.0/4.0))

    if revised:
        apar = 0.1756
        bpar = 1.0376
        cpar = 2.9787
        powr = 7.0/(2*cpar)
        hx = 1.0/gam*(1.0 - apar*xk**2)
        hx /= (1.0 + bpar*xk**2 + (apar/gam)**(1.0/powr)*xk**cpar)**powr
    else:
        aj = 0.63
        h0 = 1.0/gam
        hx = h0*(1.0 - aj*xk**2)
        fac = (h0*aj)**(4.0/7.0)
        hx /= (1.0 + fac*xk**2)**(7.0/4.0)

    isscalar=False
    if not hasattr(u,'__len__'):
        if hasattr(dv['rs'],'__len__'):
            fxcu = np.zeros(dv['rs'].shape,dtype=complex)
        else:
            isscalar=True
    else:
        fxcu = np.zeros(u.shape,dtype=complex)
    if dimensionless:
        if isscalar:
            fxcu = hx + 1.j*gx
        else:
            fxcu.real = hx
            fxcu.imag = gx
    else:
        if isscalar:
            fxcu = finf - cc*bn**(3.0/4.0)*hx -cc*bn**(3.0/4.0)*gx*1.j
        else:
            fxcu.real = finf - cc*bn**(3.0/4.0)*hx
            fxcu.imag = -cc*bn**(3.0/4.0)*gx
    return fxcu

def gki_dynamic(dv,u,axis='real',x_only=False,revised=False,param='PZ81',use_par=False):

    #if not hasattr(u,'__len__'):
    #    u = u*np.ones(1)
    if axis == 'real':
        fxcu = gki_dynamic_real_freq(dv,u,x_only=x_only,revised=revised,param=param)
    elif axis == 'imag':
        fxcu = np.zeros(u.shape)
        bn,finf = exact_constraints(dv,x_only=x_only,param=param)
        if use_par:
            if not revised and not x_only and param == 'PZ81':
                cpars = [1.06971,1.52708]#[1.06971136,1.52708142] # higher precision values destabilize the integration
                interp = 1.0/gam/(1.0 + (u.imag*bn**(0.5))**cpars[0])**cpars[1]
            elif revised and not x_only and param == 'PW92':
                cpars = [0.99711536, 1.36722527, 0.93805229, 0.0101391,  0.71194338]
                xr = u.imag*bn**(0.5)
                interp = 1.0/gam*(1.0 - cpars[3]*xr**cpars[4])/(1.0 + cpars[2]*xr**cpars[0])**cpars[1]
            fxcu = -cc*bn**(3.0/4.0)*interp + finf
        else:
            def wrap_integrand(tt,freq,rescale=False):
                if rescale:
                    alp = 0.1
                    to = 2*alp/(tt+1.0)-alp
                    d_to_d_tt = 2*alp/(tt+1.0)**2
                else:
                    to = tt
                    d_to_d_tt = 1.0
                tfxc = gki_dynamic_real_freq(dv,to,x_only=x_only,revised=revised,param=param,dimensionless=True)
                num = freq*tfxc.real + to*tfxc.imag
                denom = to**2 + freq**2
                return num/denom*d_to_d_tt
            for itu,tu in enumerate(u):
                rf = tu.imag*bn[itu]**(0.5)
                fxcu[itu],err = nquad(wrap_integrand,(-1.0,1.0),'global_adap',{'itgr':'GK','npts':5,'prec':1.e-8},args=(rf,),kwargs={'rescale':True})
                if err['error'] != err['error']:
                    fxcu[itu],err = nquad(wrap_integrand,(0.0,'inf'),'global_adap',{'itgr':'GK','npts':5,'prec':1.e-8},args=(rf,))
                if err['code'] == 0:
                    print(('WARNING, analytic continuation failed; error {:}').format(err['error']))
                fxcu[itu] = -cc*bn[itu]**(3.0/4.0)*fxcu[itu]/pi + finf[itu]
    return fxcu

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    q = np.arange(0.01,3.01,0.01)
    rsl = [1,4,10,30,69,100]
    for rs in rsl:

    #rs = 4

        dvars = {}
        dvars['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs
        dvars['rs'] = rs
        dvars['n'] = 3.0/(4*pi*rs**3)
        dvars['rsh'] = dvars['rs']**(0.5)
        f0 = alda(dvars,x_only=False,param='PW92')
        bn,finf = exact_constraints(dvars,param='PW92')
        print(rs,f0,finf,f0/finf)
        #fxc,_,_ = mcp07_static(0.5*q,q,dvars,param='PW92')
        #plt.plot(q,-fxc,label='$r_s=$'+str(rs))
    exit()
    plt.xlim([0,3])
    plt.yscale('log')
    plt.xlabel('$q/k_F$',fontsize=12)
    plt.ylabel('$-f_{xc}(q,\omega=0)$',fontsize=12)
    #plt.ylim([-500,-15.310303787092149])
    plt.legend(ncol=(len(rsl)-3))
    plt.show()
    exit()

    bn,finf = exact_constraints(dvars,param='PW92')
    #w,fxcu = np.transpose(np.genfromtxt('./rMCP07_re_fxc.csv',delimiter=',',skip_header=1))
    #fxcu = -cc*bn**(3.0/4.0)*fxcu/gam + finf
    w = np.linspace(0.0001,20.01,2000)
    fxcu = gki_dynamic(dvars,1.j*w,axis='imag',x_only=False,revised=True,param='PW92')
    #np.savetxt('./rMCP07_re_fxc.csv',np.transpose((w,fxcu)),delimiter=',',header='omega, re fxc dimensionless')
    #exit()
    """

    w,fxcu = np.transpose(np.genfromtxt('./rMCP07_re_fxc.csv',delimiter=',',skip_header=1))
    from scipy.optimize import curve_fit
    def interp(x,a,b,c,d,f):
        return( 1.0-d*x**f)/(1.0 + c*x**a)**b
    pars,cov = curve_fit(interp,w,fxcu)
    print(pars,cov)
    exit()

    import matplotlib.pyplot as plt
    fxcu = gki_dynamic(dvars,1.j*w,axis='imag',x_only=False,revised=True,param='PW92')
    plt.plot(w,fxcu)
    plt.plot(w,finf*np.ones(w.shape))
    #plt.ylim(-16,-3)
    plt.xlim(0,10)
    plt.show()
    exit()
    #"""
