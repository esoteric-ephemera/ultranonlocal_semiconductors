import numpy as np
from os import system,path
from scipy.special import gamma as gfn
from math import floor,ceil
from scipy.linalg import eigh_tridiagonal as eigtri

if not path.isdir('./grids/'):
    system('mkdir ./grids/')

def gauss_quad(lvl,grid_type='legendre',alag=0.0):

    lvl = int(np.ceil(lvl))

    if grid_type=='legendre':
        fname = 'gauss_legendre_{:}_pts.csv'.format(lvl)
    elif grid_type=='laguerre':
        fname = 'gauss_laguerre_{:}_alpha_{:}_pts.csv'.format(alag,lvl)
        if alag < -1.0:
            raise ValueError( "Generalized Laguerre grid requires alag > -1")
    elif grid_type == 'cheb':
        fname = 'gauss_cheb_{:}_pts.csv'.format(lvl)
        # NIST DLMF Eqs. 3.5.22 and 3.5.23
        k = np.arange(1,lvl+1,1)
        x = np.cos((2*k-1)*np.pi/(2.0*lvl))
        wg = np.pi/lvl*(1.0 - x**2)**(0.5)
        np.savetxt('./grids/'+fname,np.transpose((wg,x)),delimiter=',',fmt='%.16f',header='weight,point')
        return
     # algorithm from Golub and Welsch, Math. Comp. 23, 221 (1969)
    def bet(n):# coefficients from NIST's DLMF, sec. 18.9
        if grid_type == 'legendre':
            an = (2*n+1.0)/(n+1.0)
            alp = 0.0
            anp1 = (2*n+3.0)/(n+2.0)
            cnp1 = (n+1.0)/(n+2.0)
        elif grid_type=='laguerre':
            an = -1.0/(n+1.0)
            alp = -(2*n+1.0+alag)/(n+1.0)/an
            anp1 = -1.0/(n+2.0)
            cnp1 = (n+alag+1.0)/(n+2.0)
        return alp,(cnp1/an/anp1)**(0.5)
    jac = np.zeros((lvl,lvl))
    jac[0,0],jac[0,1] = bet(0)
    _,jac[-1,-2] = bet(lvl-2)
    jac[-1,-1],_ = bet(lvl-1)
    for jn in range(1,lvl-1):
        jac[jn,jn],jac[jn,jn+1] = bet(jn)
        _,jac[jn,jn-1] = bet(jn-1)
    grid,v = np.linalg.eigh(jac)
    if grid_type=='legendre':
        mu_0 = 2.0
    elif grid_type == 'laguerre':
        mu_0 = gfn(1.0+alag)

    wg = mu_0*v[0]**2

    np.savetxt('./grids/'+fname,np.transpose((wg,grid)),delimiter=',',fmt='%.16f',header='weight,point')

    return wg,grid

def gauss_kronrod(n):
    # adapted from Dirk P. Laurie,
    # CALCULATION OF GAUSS-KRONROD QUADRATURE RULE
    # Mathematics of Computation 66, 1133 (1997).
    # doi:10.1090/S0025-5718-97-00861-2
    def coeff(n):
        an = (2*n+1.0)/(n+1.0)
        alp = 0.0
        anp1 = (2*n+3.0)/(n+2.0)
        cnp1 = (n+1.0)/(n+2.0)
        return alp,(cnp1/an/anp1)#**(0.5)
    a = np.zeros(2*n+1)
    b = np.zeros(2*n+1)
    b[0]=2.0
    for jn in range(int(ceil(3*n/2.0))+1):
        if jn < int(ceil(3*n/2.0)):
            a[jn],b[jn+1] = coeff(jn)
        else:
            _,b[jn+1] = coeff(jn)
    gl_grid,gl_v = eigtri(a[:n],b[1:n]**(0.5))
    gl_wg=2.0*gl_v[0]**2
    #print(gl_grid,gl_wg)

    t = np.zeros(int(floor(n/2.0))+2)
    s = np.zeros(int(floor(n/2.0))+2)

    t[1] = b[n+1]
    for m in range(n-1):
        u = 0.0
        for k in range(int(floor((m+1.0)/2.0)),-1,-1):
            l = m-k
            u += (a[k+n+1] - a[l])*t[k+1] + b[k+n+1]*s[k] - b[l]*s[k+1]
            s[k+1] = u
        ts = s
        s = t
        t = ts
    for j in range(int(floor(n/2.0)),-1,-1):
        s[j+1] = s[j]
    for m in range(n-1,2*n-2):
        u = 0.0
        for k in range(m+1-n,int(floor((m-1.0)/2.0))+1):
            l = m - k
            j = n - 1 -l
            u += -(a[k+n+1] - a[l])*t[j+1] - b[k+n+1]*s[j+1] + b[l]*s[j+2]
            s[j+1] = u
        if m%2 == 0:
            k = int(m/2)
            a[k+n+1] = a[k] + (s[j+1] - b[k+n+1]*s[j+2])/t[j+2]
        else:
            k = int((m+1)/2)
            b[k+n+1] = s[j+1]/s[j+2]
        ts = s
        s = t
        t = ts
    a[2*n] = a[n-1] - b[2*n]*s[1]/t[1]
    grid,v = eigtri(a,b[1:]**(0.5))#
    wg = b[0]*v[0]**2

    glwg = np.zeros(wg.shape)
    for ipt,pt in enumerate(grid):
        for jp,pp in enumerate(gl_grid):
            if abs(pp-pt)<1.e-12:
                glwg[ipt] = gl_wg[jp]

    np.savetxt('./grids/'+'gauss_kronrod_'+str(2*n+1)+'_pts.csv',np.transpose((wg,grid,glwg)),delimiter=',',fmt='%.16f',header='GK weight,point, GL weight')

    return

if __name__=="__main__":

    #gauss_quad(1e3,grid_type='laguerre',alag=0.0)
    gauss_kronrod(5)
