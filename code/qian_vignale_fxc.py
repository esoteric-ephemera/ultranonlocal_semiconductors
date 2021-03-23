import numpy as np

from constants import pi
from mcp07 import exact_constraints
from integrators import nquad

def interpolate_coeffs():
    # see Table II
    rsl = np.asarray([1,2,3,4,5.0])
    al = np.asarray([0.5026,0.8473,1.092,1.278,1.426])/100.0
    bl = np.asarray([0.1555,0.1558,0.1496,0.1428,0.1363])
    gl = np.asarray([1.656,1.368,1.215,1.112,1.033])
    #ol = np.asarray([-1.484,-1.052,-0.8227,-0.6683,-0.5498])

    apars = np.polynomial.polynomial.polyfit(rsl,al,5)
    bpars = np.polynomial.polynomial.polyfit(rsl,bl,5)
    gpars = np.polynomial.polynomial.polyfit(rsl,gl,5)
    #opars = np.polynomial.polynomial.polyfit(rsl,ol,5)
    return apars,bpars,gpars#,opars

def poly(x,c):
    tmp = 0.0
    for i in range(len(c)):
        tmp += c[i]*x**i
    return tmp

def im_fxc_longitudinal(omega,rs):

    ac,bc,gc = interpolate_coeffs()
    a3 = poly(rs,ac)
    b3 = poly(rs,bc)
    g3 = poly(rs,gc)
    om3 = 1 - 1.5*g3

    wp = (3/rs**3)**(0.5)
    n = 3/(4*pi*rs**3)
    wt = omega/(2*wp)

    imfxc = a3/(1 + b3*wt**2)**(5/4)
    imfxc += wt**2*np.exp(-(np.abs(wt)-om3)**2/g3)
    imfxc *= -omega/n

    return imfxc

def wrap_kram_kron(to,omega,rs):
    return im_fxc_longitudinal(to,rs)/(to - omega)

def kram_kron(omega,rs):
    return nquad(wrap_kram_kron,('-inf','inf'),'global_adap',{'itgr':'GK','prec':1.e-6,'npts':5,'min_recur':4,'max_recur':1000,'n_extrap':400,'inf_cond':'fun'},pars_ops={'PV':[omega]},args=(omega,rs))

def fxc_longitudinal(dv,omega):

    im_fxc = im_fxc_longitudinal(omega,dv['rs'])
    _,finf=exact_constraints(dv,x_only=False,param='PZ81')
    if hasattr(omega,'__len__'):
        re_fxc = np.zeros(omega.shape)
        for iom,om in enumerate(omega):
            re_fxc[iom],terr = kram_kron(om,dv['rs'])
            if terr['code'] == 0:
                print(('WARNING, not converged for omega={:.4f}; last error {:.4e}').format(om,terr['error']))
    else:
        re_fxc,terr = kram_kron(omega,dv['rs'])
        if terr['code'] == 0:
            print(('WARNING, not converged for omega={:.4f}; last error {:.4e}').format(om,terr['error']))
    return re_fxc/pi + finf + 1.j*im_fxc


if __name__=="__main__":
    rs = 3
    dvars = {}
    dvars['kF'] = (9.0*pi/4.0)**(1.0/3.0)/rs
    dvars['rs'] = rs
    dvars['n'] = 3.0/(4.0*pi*dvars['rs']**3)
    dvars['rsh'] = dvars['rs']**(0.5)

    wp = (3/rs**3)**(0.5)

    om = wp*np.linspace(0.0,5.0,50)
    fxc_qv = fxc_longitudinal(dvars,om)*dvars['n']/(2*wp)
    import matplotlib.pyplot as plt
    plt.plot(om/wp,fxc_qv.imag)
    plt.plot(om/wp,fxc_qv.real)
    plt.show()
