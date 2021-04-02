import numpy as np

from constants import pi
from mcp07 import exact_constraints,alda
from integrators import nquad

"""
    Zhixin Qian and Giovanni Vignale,
    ``Dynamical exchange-correlation potentials for an electron liquid'',
    Phys. Rev. B 65, 235121 (2002).
    https://doi.org/10.1103/PhysRevB.65.235121
"""

# [Gamma(1/4)]**2
gamma_14_sq = 13.1450472065968728685447786119766533374786376953125

def density_variables(rs):
    dv = {}
    dv['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs
    dv['rs'] = rs
    dv['n'] = 3.0/(4.0*pi*dv['rs']**3)
    dv['rsh'] = dv['rs']**(0.5)
    dv['wp0'] = (3/rs**3)**(0.5)
    return dv

def mu_xc_per_n(rs,a,b,c):
    return a/rs + (b-a)*rs/(rs**2 + c)

def fit_mu_xc():
    # Data from Table 1 of QV, mu_xc in units of 2 omega_p(0) n
    mul = np.asarray([[1, 0.0073], [2, 0.0077], [3, 0.00801], [4, 0.00837], [5, 0.00851]])
    # convert to mu_xc/n
    mul[:,1] *= 2*(3/mul[:,0]**3)**(0.5)
    from scipy.optimize import curve_fit
    opts,_ = curve_fit(mu_xc_per_n,mul[:,0], mul[:,1])
    for opt in opts:
        print(opt)
    return

def erf(x):
    """   Abramowitz and Stegun eq. 7.1.26
        maximum absolute error is 1.5 x 10**(-7)
    """
    p = 0.3275911
    a = [0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429]
    """ erf(-x) = - erf(x)  """
    sgnx = np.sign(x)
    t = 1/(1 + p*sgnx*x)
    ttmp = np.ones(t.shape)
    tmp = np.zeros(t.shape)
    for i in range(5):
        ttmp *= t
        tmp += a[i]*ttmp
    return sgnx*(1 - tmp*np.exp(-x**2))

def bracket(fun,bds,nstep=500,vector=False,args=(),kwargs={}):

    step = (bds[1]-bds[0])/nstep
    ivals = []
    if vector:
        tmpl = np.arange(bds[0],bds[1],step)
        funl = fun(tmpl,*args,**kwargs)
        ofun = funl[0]
        for istep in range(1,nstep):
            if ofun*funl[istep] <= 0:
                ivals.append([tmpl[istep-1],tmpl[istep]])
            ofun = funl[istep]
    else:
        tmp = bds[0]
        for istep in range(nstep):
            cfun = fun(tmp,*args,**kwargs)
            if istep == 0:
                ofun = cfun
            if ofun*cfun <= 0:
                ivals.append([tmp-step,tmp])
            ofun = cfun
            tmp += step
    return ivals

def bisect(fun,bds,tol=1.e-6,maxstep=100,args=(),kwargs={}):

    """
        Routine adapted from
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery,
        ``Numerical Recipes in Fortran 77:
        The Art of Scientific Computing'',
        2nd ed., Cambridge University Press 1992
        ISBN 0-521-43064-X
    """

    def wrap_fun(arg):
        return fun(arg,*args,**kwargs)
    tmp0 = wrap_fun(bds[0])
    tmp1 = wrap_fun(bds[1])
    if tmp0*tmp1 > 0:
        raise ValueError('No root in bracket')
    if tmp0 < 0.0:
        regs = [bds[0],bds[1]]
    else:
        regs = [bds[1],bds[0]]
    for iter in range(maxstep):
        mid = (regs[0] + regs[1])/2.0
        spc = mid - regs[0]
        tmpm = wrap_fun(mid)
        if tmpm < 0.0:
            regs[0] = mid
        else:
            regs[1] = mid
        if abs(spc) < tol*abs(mid) or abs(tmpm)<tol:
            break
    return mid,tmpm

def s_3_l(kf):
    """ Eq. 16 of Qian and Vignale """
    lam = (pi*kf)**(0.5) # lambda = 2 k_F/ k_TF
    s3l = 5 - (lam + 5/lam)*np.arctan(lam) - 2/lam*np.arcsin(lam/(1+lam**2)**(0.5))
    s3l += 2/(lam*(2+lam**2)**(0.5))*(pi/2 - np.arctan(1/(lam*(2+lam**2)**(0.5))))
    return -s3l/(45*pi)

def get_qv_pars(dv,use_mu_xc=True):

    c3l = 23/15 # just below Eq. 13
    s3l = s_3_l(dv['kF'])
    """     Eq. 28   """
    a3l = 2*(2/(3*pi**2))**(1/3)*dv['rs']**2*s3l
    """     Eq. 29   """
    b3l = 16*(2**10/(3*pi**8))**(1/15)*dv['rs']*(s3l/c3l)**(4/5)

    fxc_0 = alda(dv,x_only=False,param='PW92')
    _,fxc_inf=exact_constraints(dv,x_only=False,param='PW92')

    # approx. expression for mu_xc that interpolates metallic data, from Table 1 of QV
    # fitting code located in fit_mu_xc
    if use_mu_xc:
        mu_xc_n = mu_xc_per_n(dv['rs'],0.03115158529677855,0.011985054514894128,2.267455018224077)
        fxc_0 += 4/3*mu_xc_n/dv['n']

    df = fxc_0-fxc_inf

    def solve_g3l(tmp):

        """    Eq. 27    """
        o3l = 1 - 1.5*tmp
        o3l2 = o3l**2
        """    Eq. 30    """
        res = 4*(2*pi/b3l)**(0.5)*a3l/gamma_14_sq
        res += o3l*tmp*np.exp(-o3l2/tmp)/pi + 0.5*(tmp/pi)**(0.5)*(tmp + 2*o3l2)*(1 + erf(o3l/tmp**(0.5)))
        res *= 4*(pi/dv['n'])**(0.5) # 2*omega_p(0)/n
        return df + res

    poss_brack = bracket(solve_g3l,(1.e-6,3.0),nstep=500,vector=True)
    g3l = 1.e-14
    for tbrack in poss_brack:
        tg3l,success = bisect(solve_g3l,tbrack,tol=1.5e-7,maxstep=200)
        if success < 1.5e-7:
            g3l = max(tg3l,g3l)
    o3l = 1 - 1.5*g3l
    return a3l,b3l,g3l,o3l

def im_fxc_longitudinal(omega,dv):

    a3,b3,g3,om3 = get_qv_pars(dv)

    wt = omega/(2*dv['wp0'])

    imfxc = a3/(1 + b3*wt**2)**(5/4)
    imfxc += wt**2*np.exp(-(np.abs(wt)-om3)**2/g3)
    imfxc *= -omega/dv['n']

    return imfxc

def wrap_kram_kron(to,omega,dv):
    return im_fxc_longitudinal(to,dv)/(to - omega)

def kram_kron(omega,dv):
    return nquad(wrap_kram_kron,('-inf','inf'),'global_adap',{'itgr':'GK','prec':1.e-6,'npts':5,'min_recur':4,'max_recur':1000,'n_extrap':400,'inf_cond':'fun'},pars_ops={'PV':[omega]},args=(omega,dv))

def fxc_longitudinal(dv,omega):

    im_fxc = im_fxc_longitudinal(omega,dv)
    _,finf=exact_constraints(dv,x_only=False,param='PW92')
    if hasattr(omega,'__len__'):
        re_fxc = np.zeros(omega.shape)
        for iom,om in enumerate(omega):
            re_fxc[iom],terr = kram_kron(om,dv)
            if terr['code'] == 0:
                print(('WARNING, not converged for omega={:.4f}; last error {:.4e}').format(om,terr['error']))
    else:
        re_fxc,terr = kram_kron(omega,dv)
        if terr['code'] == 0:
            print(('WARNING, not converged for omega={:.4f}; last error {:.4e}').format(omega,terr['error']))
    return re_fxc/pi + finf + 1.j*im_fxc


if __name__=="__main__":

    fit_mu_xc()
    exit()

    for rs in [2.00876975652943]:#[0.1,0.5,1,2,3,4,5,10,20,30,40,69,100]:
        dv = density_variables(rs)
        print(exact_constraints(dv,x_only=False,param='PW92'))
        print(exact_constraints(dv,x_only=False,param='PZ81'))
        print(rs,get_qv_pars(rs))
    exit()

    rs = 3
    dvars = density_variables(rs)

    wp = (3/rs**3)**(0.5)

    om = wp*np.linspace(0.0,5.0,50)
    fxc_qv = fxc_longitudinal(dvars,om)*dvars['n']/(2*wp)
    import matplotlib.pyplot as plt
    plt.plot(om/wp,fxc_qv.imag)
    plt.plot(om/wp,fxc_qv.real)
    plt.show()
