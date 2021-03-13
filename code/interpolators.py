import numpy as np

# Spline interpolation from W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery
# Numerical Recipes in Fortran 77: The Art of Scientific Computing
# 2nd edn., Vol. 1, Cambridge University Press, 1992

def natural_spline(x,y):

    u = np.zeros(x.shape)
    y2 = np.zeros(x.shape)
    for i in range(1,x.shape[0]-1):
        s = (x[i] - x[i-1]) / (x[i+1] - x[i-1])
        p = s*y2[i-1] + 2.0
        y2[i] = (s - 1.0)/p
        u[i] = (6.0*((y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]) )/(x[i+1]-x[i-1]) - s*u[i-1])/p
    # natural spline only, y'' = 0 on boundary
    y2[-1] = 0.0
    for i in range(x.shape[0]-2,-1,-1):
        y2[i] = y2[i]*y2[i+1] + u[i]
    return y2

def spline(x,xf,yf,yf2):
    # NB: yf2 is result of natural_spline
    if not hasattr(yf2,'__len__') and yf2 == None: # maybe we only need a one-shot spline, and not repeated calls to spline
        yf2 = natural_spline(xf,yf) # if that's the case, grab the 2nd derivatives
    hi = np.searchsorted(xf,x,side='left')
    lo = hi-1
    h = xf[hi] - xf[lo]
    a = (xf[hi] - x)/h
    b = (x - xf[lo])/h
    y = a*yf[lo] + b*yf[hi]
    y += ( (a**3 - a)*yf2[lo] + (b**3 - b)*yf2[hi])*h**2/6.0

    return y

def lagrange_interp(x,xf,yf):
    interp = np.zeros(xf.shape)#,dtype=yf.dtype)
    ty = np.zeros(x.shape)#,dtype=yf.dtype)
    truth_array = np.ones(xf.shape,dtype=bool)
    for itx,tx in enumerate(x):
        for ind in range(xf.shape[0]):
            truth_array[ind] = False
            num = tx-xf[truth_array]
            denom = xf[ind]-xf[truth_array]
            interp[ind] = yf[ind]*np.prod(num/denom)
            truth_array[ind] = True
        ty[itx] = np.sum(interp)
    return ty

def linear_interpolator(x,x_tab,y_tab):
    # vectorized linear interpolation
    hi = np.searchsorted(x_tab,x,side='left')
    slope = (y_tab[hi]-y_tab[hi-1])/(x_tab[hi] - x_tab[hi-1])
    return slope*(x-x_tab[hi-1]) + y_tab[hi-1]

if __name__=="__main__":

    kc_d = {4.0: 2.97, 69.0: 3.92}#6.65}#

    for rs in [4.0,69.0]:
        x = np.linspace(0.1,3.0,300)
        q,sq = np.transpose(np.genfromtxt('./interp/Sq_'+str(rs)+'_0.01spc_tab.csv',delimiter=',',skip_header=1))
        #q,sq = np.transpose(np.genfromtxt('./interp/Sq_'+str(rs)+'_0.01spc_tab.csv',delimiter=',',skip_header=1))
        #ts=l_interp(x,q,sq)
        #ts2 = natural_spline(q,sq)
        ts = spline(x,q,sq,None)
        np.savetxt('testfile_'+str(rs)+'.csv',np.transpose((x,ts)),delimiter=',')
