import numpy as np
from math import floor,ceil
from os import path
from gauss_quad import gauss_kronrod,gauss_quad
from itertools import product
from interpolators import lagrange_interp

"""
Author: Aaron Kaplan
"""

pi = np.pi

################################################################################

def nquad(fun,ttbd,method,opts,pars_ops={},args=(),kwargs={}):
    """
    nquad_core is the primary routine here
    ----------------------------------------------------------------------------
    fun is the function to be integrated
    this assumes the form fun(x,*args,**kwargs), where x is the integration variable
    args are any normal arguments, and kwargs are keyword/dictionary arguments.

    See the bottom of this routine for examples of use
    ----------------------------------------------------------------------------
    ttbd are the bounds, in the form (lower bound, upper bound)
    To have an improper bound, set lower bound = '-inf', upper bound = 'inf',
    or both
    ----------------------------------------------------------------------------
    method can be:
        'global_adap': global-adaptive quadrature (recommended by default)
            Default method is 11-point Gauss-Kronrod ('GK')
            Can also select doubly-adaptive Clenshaw-Curtis quadrature ('CC')

            use 'itgr' in opts to set the quadrature mesh
            use 'npts' in opts to set the number of points,
                but NB, the actual number of grid points will be 2*'npts'+1
                for both GK and CC

        'trap': standard adaptive trapezoid integration method
            use 'simpson'=True in opts to use the adaptive Simpson quadrature
            instead

        'clenshaw_curtis': singly-adaptive Clenshaw-Curtis quadrature. Very similar
            to the trapezoidal rule, except the weights and points are chosen by
            Clenshaw-Curtis quadrature.

        'dbl_exp': the family of double-exponential integration methods, also
            sometimes called tanh-sinh integration. A trapezoidal rule common
            amongst mathematicians
    ----------------------------------------------------------------------------
    use 'prec' in opts to set the precision for any of the routines

    different methods have different options
    ----------------------------------------------------------------------------
    pars_ops is a dictionary used to control the behavior of nquad:

        'inf_cond' is the method used to determine the integration bound when
        one or both of the bounds are +- infinity

            'fun_and_deriv' tries to set the function and deriviative equal to zero
                This is the default, and recommended for 1D integrals

            'fun' is much coarser and only attempts to minimize the function

            'integral' simply partitions the integration domain into disjoint
                subregions, and adds them until the value of the integral converges.
                In this method, each subregion is converged to the error tolerance.
                Recommended for use in higher-dimensional integrals, as this
                uses the least number of function evaluations

        'PV' allows for Cauchy principal value integrals. The entry for 'PV'
        should be a list.

        'PV_eps' sets the tolerance around each singularities in a PV integral

        'min_ubd' can enforce a minimum upper bound whenever the upper bound is 'inf'

        'min_lbd' does the same whenever the lower bound is '-inf'

    ----------------------------------------------------------------------------
    """
    def nquad_core(bds):

        if method!='dbl_exp':

            if 'inf_cond' not in pars_ops:
                pars_ops['inf_cond'] = 'fun_and_deriv'
            if 'n_extrap' in pars_ops:
                max_inc = pars_ops['n_extrap']
            else:
                max_inc = 200

            if bds[0] == '-inf' or bds[1] == 'inf':
                def fun_inv(x):
                    return 1.0/x**2*fun(1.0/x,*args,**kwargs)
                if pars_ops['inf_cond'] != 'integral':
                    def wrap_fun(x):
                        return fun(x,*args,**kwargs)

            if bds[0] =='-inf':
                if pars_ops['inf_cond'] != 'integral':
                    lbd = get_inf_bounds(pars_ops['inf_cond'],'lower',bds,targf=wrap_fun,max_inc=max_inc)
                if 'min_lbd' in pars_ops:
                    lbd = min(lbd,pars_ops['min_lbd'])
            else:
                lbd = bds[0]

            if bds[1] =='inf':
                if pars_ops['inf_cond'] != 'integral':
                    ubd = get_inf_bounds(pars_ops['inf_cond'],'upper',bds,targf=wrap_fun,max_inc=max_inc)
                if 'min_ubd' in pars_ops:
                    ubd = max(ubd,pars_ops['min_ubd'])
            else:
                ubd = bds[1]

        if method=='global_adap':

            if bds[0]=='-inf' or bds[1] == 'inf':
                int_l = 0.0
                errd_l = {'code':1,'error':0.0}
                int_u = 0.0
                errd_u = {'code':1,'error':0.0}
                def wrap_itgr(tbd):
                    return global_adap(fun,tbd,opts,args=args,kwargs=kwargs)
                def wrap_itgr_inv(tbd):
                    return global_adap(fun_inv,tbd,opts)

            if bds[0] == '-inf':
                if pars_ops['inf_cond'] == 'integral':
                    lbd,int_l,errd_l = get_inf_bounds(pars_ops['inf_cond'],'lower',bds,itgr=wrap_itgr,max_inc=max_inc)
                    int_l,errd_l = extrapolate_to_inf(lbd,'lower',int_l,errd_l,wrap_itgr_inv)
                else:
                    if bds[1] != 'inf': # if the full range is -infinity to infinity, wait to calculate integral
                        int_l,errd_l=global_adap(fun,(lbd,ubd),opts,args=args,kwargs=kwargs)
                        int_l,errd_l = extrapolate_to_inf(lbd,'lower',int_l,errd_l,wrap_itgr_inv)

            if bds[1]=='inf':
                if pars_ops['inf_cond'] == 'integral':
                    ubd,int_u,errd_u = get_inf_bounds(pars_ops['inf_cond'],'upper',bds,itgr=wrap_itgr,max_inc=max_inc)
                    int_u,errd_u = extrapolate_to_inf(ubd,'upper',int_u,errd_u,wrap_itgr_inv)
                else: # regardless of lower bound, we now have it, and can do the full integral
                    int_u,errd_u=global_adap(fun,(lbd,ubd),opts,args=args,kwargs=kwargs)
                    int_u,errd_u = extrapolate_to_inf(ubd,'upper',int_u,errd_u,wrap_itgr_inv)
                    if bds[0] == '-inf':
                        int_u,errd_u = extrapolate_to_inf(lbd,'lower',int_u,errd_u,wrap_itgr_inv)

            if bds[0] == '-inf' or bds[1] == 'inf':
                oint = int_l + int_u
                oerr = {'code': min([errd_l['code'],errd_u['code']]),'error': errd_l['error']+errd_u['error']}
            else:
                oint,oerr=global_adap(fun,(lbd,ubd),opts,args=args,kwargs=kwargs)
            return oint,oerr

        elif method=='trap':
            int,err=trap(fun,(lbd,ubd),opts,args=args,kwargs=kwargs)
            if bds[1]=='inf':
                int2,err2=trap(fun_inv,(1.e-10,1.0/ubd),opts)
                if err['code']==1 or err2['code']==1:
                    errd={'code': 1,'error':( err['error']+err2['error'])/2.0 }
                else:
                    errd={'code': 0,'error': ( err['error']+err2['error'])/2.0 }
                return int+int2,errd
            else:
                return int,err
        elif method == 'clenshaw_curtis':
            int,err=adap_cc(fun,(lbd,ubd),opts,args=args,kwargs=kwargs)
            if bds[1]=='inf':
                int2,err2=adap_cc(fun_inv,(1.0/ubd**2,1.0/ubd),opts)
                if err['code']==1 or err2['code']==1:
                    errd={'code': 1,'error':( err['error']+err2['error'])/2.0 }
                else:
                    errd={'code': 0,'error': ( err['error']+err2['error'])/2.0 }
                return int+int2,errd
            else:
                return int,err
        elif method=='dbl_exp':
            return dbl_exp(fun,bds,opts,args=args,kwargs=kwargs)
#-------------------------------------------------------------------------------

    nreg = 1
    if ttbd[0] == '-inf':
        nreg += 1
    if ttbd[1] == 'inf':
        nreg += 1

    if 'PV' in pars_ops: # for handling numeric principal value integrals
        if 'prec' in opts:
            opts['prec'] /= nreg + len(pars_ops['PV'])
        else:
            opts['prec']=1.e-6/(nreg + len(pars_ops['PV']))
        eps = 1.e-12
        if 'PV_eps' in pars_ops: # if the tolerance on epsilon is too tight, or too loose
            eps = pars_ops['PV_eps'] # can be set via pars_ops
        intf = 0.0
        errf = {'code':1, 'error': 0.0}
        for itt in range(len(pars_ops['PV'])+1): # can handle arbitrary number of singularities
            if itt == 0:
                rng = (ttbd[0],pars_ops['PV'][itt] - eps)
            elif itt == len(pars_ops['PV']):
                rng = (pars_ops['PV'][itt-1] + eps,ttbd[1])
            else:
                rng = (pars_ops['PV'][itt-1] + eps,pars_ops['PV'][itt] - eps)
            tint,terr = nquad_core(rng)
            intf += tint
            errf['code'] = min([errf['code'],terr['code']])
            errf['error'] += terr['error']

    else: # for handling everything else
        if 'prec' in opts:
            opts['prec']/= nreg
        else:
            opts['prec']=1.e-6/nreg
        intf,errf = nquad_core(ttbd)

    return intf,errf


################################################################################

def get_inf_bounds(method,wbound,bds,targf=None,itgr=None,max_inc=1000):

    if wbound == 'lower':
        sgn = -1.0
        if bds[1]== 'inf' or bds[1] > 0.0:
            sbd = 0.0
        else:
            sbd = bds[1]
    elif wbound == 'upper':
        sgn = 1.0
        if bds[0] == '-inf' or bds[0] < 0.0:
            sbd = 0.0
        else:
            sbd = bds[0]
    if sbd == 0.0:
        sbd += sgn*1.e-4

    if method != 'integral':
        bd = sbd
        oval = targf(np.asarray([bd]))
        relval= min([1.0,abs(oval)])
        for ival in range(3):
            for inc in range(max_inc):
                bd += sgn*10.0**(-ival)
                tval = targf(np.asarray([bd]))
                if method == 'fun_and_deriv':
                    prog_cond = abs(tval)<10.0**(-4-ival)*relval and abs(oval-tval) < 10.0**(-4-ival)*relval
                elif method == 'fun':
                    prog_cond = abs(tval)<10.0**(-1-ival)*relval
                if prog_cond:
                    if ival < 2:
                        if wbound == 'lower':
                            bd = min([sbd,bd - sgn*10.0**(-ival)])
                        elif wbound == 'upper':
                            bd = max([sbd,bd - sgn*10.0**(-ival)])
                    break
                else:
                    oval = tval
                    if abs(tval)>relval:
                        relval = abs(tval)
        return bd
    elif method == 'integral':
        errd = {'code': 1, 'error': 0.0}
        int = 0.0
        bd = sbd
        for ival in range(3):
            for inc in range(max_inc):
                lower = bd
                bd += sgn*10.0**(-ival)
                tsum,err = itgr((lower,bd))
                if ival == 0 and inc == 0:
                    init = abs(tsum)
                if abs(tsum) < init*10**(-3-ival):
                    if ival < 2:
                        if wbound == 'lower':
                            bd = min([sbd,bd - sgn*10.0**(-ival)])
                        elif wbound == 'upper':
                            bd = max([sbd,bd - sgn*10.0**(-ival)])
                    else:
                        int += sgn*tsum
                        errd['code'] = min([errd['code'],err['code']])
                        errd['error'] +=  err['error']
                    break
                else:
                    int += sgn*tsum
                    errd['code'] = min([errd['code'],err['code']])
                    errd['error'] += err['error']
        return bd,int,errd

def extrapolate_to_inf(bd,wbound,int,errd,itgr):

    if wbound == 'lower':
        sgn = -1.0
    elif wbound == 'upper':
        sgn = 1.0

    if abs(bd) >= 1.0:
        npt = sgn/(1.0 + bd**2)
    else:
        npt = sgn*np.exp(-abs(bd))
    mid = (npt + 1.0/bd)/2.0
    int2,err2 = itgr((npt,1.0/bd))
    int3,err3 = itgr((mid,1.0/bd))
    int2 *= sgn
    int3 *= sgn
    if err2['code']==1 and err3['code']==1:
        int_inf = lagrange_interp(np.zeros(1),np.asarray([mid,npt]),np.asarray([int3,int2]))[0]
        errd['error'] += (err2['error']+err3['error'])/2.0
    elif err2['code']==1 and err3['code']==0:
        int_inf = int2
        errd['error'] += err2['error']
    elif err2['code']==0 and err3['code']==1:
        int_inf = int3
        errd['error'] += err3['error']
    else:
        int_inf = 0.0
    if abs(int_inf)>abs(int):
        int_inf = 0.0
    return int+int_inf,errd

################################################################################

def global_adap(fun,bds,opt_d,args=(),kwargs={}):

    """
        error codes:
        > 0   successful integration
            1    absolutely no issues
        <= 0   unsucessful integration:
            0   exceeded maximum number of steps
           -1   NaN error (errors are NaN)
           -2   Bisection yielded regions smaller than machine precision
           -3   Result was below machine precision, estimating as zero
    """
    meps = 7/3-4/3-1 # machine precision

    lbd,ubd = bds

    def_pts = {'GK': 5, 'CC': 12}
    prec = 1.0e-6
    if 'prec' in opt_d:
        prec = opt_d['prec']
    min_recur = 2
    if 'min_recur' in opt_d:
        min_recur = opt_d['min_recur']

    if 'max_recur' in opt_d:
        max_recur = opt_d['max_recur']
    else:
        # 2**max_recur bisections yields a region of width 10**(-60)
        max_recur = ceil((np.log(abs(bds[1]-bds[0])) + 60*np.log(10.0))/np.log(2.0))
        max_recur = max(max_recur,1000)

    if 'itgr' not in opt_d:
        opt_d['itgr'] = 'GK'
    if 'npts' in opt_d:
        npts = opt_d['npts']
    else:
        npts = def_pts[opt_d['itgr']]
    if 'error monitoring' not in opt_d:
        opt_d['error monitoring'] = False
    if 'err_meas' not in opt_d:
        opt_d['err_meas'] = 'abs_diff'

    if 'rel_tol' in opt_d:
        rel_tol = opt_d['rel_tol']
    else:
        rel_tol = min(0.01,100*prec)

    if opt_d['itgr'] == 'GK':
        def_grid = './grids/gauss_kronrod_'+str(2*npts+1)+'_pts.csv'
        if not path.isfile(def_grid) or path.getsize(def_grid)==0:
            gauss_kronrod(npts) # returns 2*N + 1 points
        wg,mesh,wg_err = np.transpose(np.genfromtxt(def_grid,delimiter=',',skip_header=1))
    elif opt_d['itgr'] == 'CC' :
        def_grid = './grids/clenshaw_curtis_'+str(2*npts+1)+'.csv'
        if not path.isfile(def_grid) or path.getsize(def_grid)==0:
            clenshaw_curtis_grid(npts) # returns 2*N + 1 points
        mesh,wg,wg_err = np.transpose(np.genfromtxt(def_grid,delimiter=',',skip_header=1))

    if 'reg' in opt_d:
        working_regs = []
        for iareg,areg in enumerate(opt_d['reg']):
            if iareg == 0:
                working_regs.append([lbd,areg[1]])
            elif iareg == len(opt_d['reg'])-1:
                working_regs.append([areg[0],ubd])
            else:
                working_regs.append(areg)
    else:
        treg = np.linspace(lbd,ubd,min_recur+1)
        working_regs = []
        for ibord in range(len(treg)-1):
            working_regs.append([treg[ibord],treg[ibord+1]])

    reg_l = np.zeros((0,2))
    err_l = np.zeros(0)
    sum_l = np.zeros(0)

    for irecur in range(max_recur):

        for areg in working_regs:
            x_mesh = 0.5*(areg[1]-areg[0])*mesh + 0.5*(areg[1]+areg[0])
            x_wg = 0.5*(areg[1]-areg[0])*wg
            x_wg_err = 0.5*(areg[1]-areg[0])*wg_err
            tvar = fun(x_mesh,*args,**kwargs)

            tint = np.sum(x_wg*tvar)
            tint_gl = np.sum(x_wg_err*tvar)

            reg_l = np.vstack((reg_l,areg))
            sum_l = np.append(sum_l,tint)
            if opt_d['err_meas']=='quadpack':
                """
                empirical error measure from:
                R. Piessens, E. de Doncker-Kapenga,  C. W. Uberhuber, and D. K. Kahaner
                ``QUADPACK: A Subroutine Package for Automatic Integration''
                Springer-Verlag, Berlin, 1983.
                doi: 10.1007/978-3-642-61786-7
                """
                fac = np.sum(x_wg*np.abs(tvar - tint/(areg[1]-areg[0])))
                gk_err = abs(tint-tint_gl)
                if fac == 0.0:
                    lerr_meas = 0.0
                else:
                    lerr_meas = fac*min(1.0,(200*gk_err/fac)**(1.5))
                err_l = np.append(err_l,lerr_meas)
            elif opt_d['err_meas']=='abs_diff' or opt_d['err_meas']=='global_rel':
                err_l = np.append(err_l,abs(tint-tint_gl))
            elif opt_d['err_meas']=='local_rel':
                err_l = np.append(err_l,abs(tint-tint_gl)/max(meps,abs(tint)))

        csum = np.sum(sum_l)
        cprec = max(meps,min(prec,abs(csum)/2))

        if opt_d['err_meas']=='global_rel':
            global_error = np.sum(err_l)/max(meps,csum)
        else:
            global_error = np.sum(err_l)

        if opt_d['error monitoring']:
            print(global_error,csum)

        if abs(csum)< meps:
            return 0.0,{'code':-3,'error':global_error}

        if global_error != global_error: # if the output is NaN, completely failed
            return csum,{'code':-1,'error':global_error}

        if global_error < cprec: # SUCCESS!!!!
            return csum,{'code':1,'error':global_error}
        else:
            inds = np.argsort(err_l)
            bad_reg = reg_l[inds][-1]
            bad_err = err_l[inds][-1]
            err_l = err_l[inds][:-1]
            reg_l = reg_l[inds][:-1]
            sum_l = sum_l[inds][:-1]
            if opt_d['itgr'] == 'CC' and np.sum(err_l) < 0.8*prec:
                tmp_l = fun(0.5*(bad_reg[1]-bad_reg[0])*mesh + 0.5*(bad_reg[1]+bad_reg[0]),*args,**kwargs)
                tmp_w = 0.5*(bad_reg[1]-bad_reg[0])*wg
                for it in range(0,2):
                    tmp_l,tmp_w,tmp_werr=increase_cc_grid(fun,(bad_reg[0],bad_reg[1]),tmp_l,tmp_w,args=args,kwargs=kwargs)
                    tsum = np.sum(tmp_w*tmp_l)
                    terr = np.sum(tmp_werr*tmp_l)
                    if opt_d['err_meas']=='quadpack':
                        fac = np.sum(tmp_w*np.abs(tmp_l - tsum/(bad_reg[1]-bad_reg[0])))
                        gk_err = abs(tsum-terr)
                        if fac == 0.0:
                            err = 0.0
                        else:
                            err = fac*min(1.0,(200*gk_err/fac)**(1.5))
                    elif opt_d['err_meas']=='abs_diff' or opt_d['err_meas']=='global_rel':
                        err = abs(tsum-terr)
                    elif opt_d['err_meas']=='local_rel':
                        err=abs(tint-tint_gl)/max(1.e-14,abs(tint))

                    if opt_d['err_meas']=='global_rel':
                        new_global_error = (np.sum(err_l)+err)/(csum+tsum)
                    else:
                        new_global_error = np.sum(err_l)+err

                    if new_global_error < cprec:
                        return csum+tsum,{'code':1,'error':new_global_error}
                    else:
                        if err > bad_err:
                            break
                        else:
                            bad_err = err

            mid = (bad_reg[0] + bad_reg[1])/2.0 # recursive bisection of highest error region
            if abs(bad_reg[1]-bad_reg[0])< meps or abs(bad_reg[1]-mid)< meps or abs(bad_reg[0]-mid)< meps:
                # bisection yields differences below machine precision, integration failed
                return csum,{'code':-2,'error':global_error}
            working_regs = [[bad_reg[0],mid],[mid,bad_reg[1]]]

    if irecur == max_recur-1:
        if abs(csum)<meps:
            return 0.0,{'code':-3,'error':global_error}
        else:
            return csum,{'code':0,'error':global_error}


################################################################################

def trap(fun,bds,opt_d,args=(),kwargs={}):
    prec=1.e-10
    if 'prec' in opt_d:
        prec = opt_d['prec']
    simpson=False
    if 'simpson' in opt_d:
        simpson = opt_d['simpson']
    h = (bds[1]-bds[0])
    max_depth = 40 # minimum step size is 2**(max_depth+1)
    min_depth = 2
    prev_h_int = -1e20
    tsum = 0.0
    otsum = -1e20

    for iter in range(max_depth+1):

        if iter == 0:
            m_l = np.asarray([bds[0],bds[1]])
            tsum += 0.5*h*np.sum(fun(m_l,*args,**kwargs))
        else:
            m_l = np.arange(bds[0]+h,bds[1],2*h)
            tsum += h*np.sum(fun(m_l,*args,**kwargs))

        if simpson:
            ttsum = tsum
            tsum = (4*ttsum - otsum)/3.0

        if abs(prev_h_int - tsum) < prec and iter > min_depth-1:
            return tsum,{'code':1,'error': abs(prev_h_int - tsum) }
        else:
            l_err = abs(prev_h_int - tsum)
            prev_h_int = tsum
            if simpson:
                otsum = ttsum
                tsum = ttsum/2.0
            else:
                tsum /= 2.0 # reuse previous integrated value
            h/=2.0 # halve the step size

    if iter==max_depth:
        return tsum,{'code':0,'error': l_err }

################################################################################

def dbl_exp(fun,bds,opt_d,args=(),kwargs={}):

    #  H.  Takahasi and M. Mori,
    # ``Double Exponential Formulas for Numerical Integration''
    # Publications of the Research Institute for Mathematical Sciences 9, 721-741 (1974).
    # DOI: 10.2977/prims/119519245

    h = 0.5
    min_depth = 2
    max_depth = 40 # minimum step size is 2**(max_depth+1)

    prev_h_int = -1e20
    tsum = 0.0
    prec = 1e-6
    if 'prec' in opt_d:
        prec = opt_d['prec']

    if bds[0] != '-inf' and bds[1] == 'inf':
        typ = 'semi_inf_pos'
    elif bds[0] == '-inf' and bds[1] != 'inf':
        typ = 'semi_inf_neg'
    elif bds[0] == '-inf' and bds[1] == 'inf':
        typ = 'dbl_inf'
    else:
        h*=(bds[1]-bds[0])
        typ = 'finite'

    def change_vars(u):
        if typ == 'semi_inf_pos':
            phi = np.exp(pi/2.0*np.sinh(u)) + bds[0]
            dphi = pi/2.0*phi*np.cosh(u)
        elif typ == 'semi_inf_neg':
            phi = -np.exp(pi/2.0*np.sinh(u)) + bds[1]
            dphi = pi/2.0*phi*np.cosh(u)
        elif typ == 'dbl_inf':
            phi = np.sinh(pi/2.0*np.sinh(u))
            dphi = pi/2.0*np.cosh(pi/2.0*np.sinh(u))*np.cosh(u)
        else:
            phi = np.tanh(pi/2.0*np.sinh(u))
            dphi = pi/2.0*np.cosh(u)/np.cosh(pi/2.0*np.sinh(u))**2
            phi = 0.5*(bds[1]-bds[0])*phi + 0.5*(bds[1]+bds[0])
            dphi *= 0.5*(bds[1]-bds[0])
        return phi,dphi

    for iter in range(max_depth+1):

        step = 2 # want to reuse previous integrated value
        if iter == 0: # as h is halved in each iteration,
            step = 1 # only use odd values of m*h after initial iteration
        if typ == 'semi_inf_pos' or typ == 'semi_inf_neg':
            max_m = ceil(np.arcsinh(2.0/pi*50*np.log(10.0))/h)
        elif typ == 'dbl_inf':
            max_m = ceil(np.arcsinh(2.0/pi*50*np.arcsinh(10.0))/h)
        else:
            max_m = ceil(np.arcsinh(10.0**30)/h)

        for sum_fac in [-1,1]:

            lbd = 1
            if iter == 0 and sum_fac == -1:
                lbd = 0

            m_conv = False
            oint = -1e20
            m = 0
            tint = 0
            for m in range(lbd,max_m+1,step):

                p,dp = change_vars(sum_fac*m*h)
                tint += fun(p,*args,**kwargs)*dp*h

                if abs(tint - oint) < prec:
                    tsum += tint
                    m_conv = True
                    break
                else:
                    l_err = abs(tint - oint)
                    oint = tint

            if not m_conv:
                print('WARNING: double exponential quadrature not converged; m summation')
                if l_err != l_err:
                    raise SystemExit("FAILURE: Error returned NaN!")

        if abs(prev_h_int - tsum) < prec and iter >= min_depth:
            return tsum,{'code':1,'error': abs(prev_h_int - tsum) }
        else:
            l_err = abs(prev_h_int - tsum)
            prev_h_int = tsum
            tsum /= 2.0 # reuse previous integrated value
            h/=2.0 # halve the step size

    if iter==max_depth:
        return tsum,{'code':0,'error': l_err }


################################################################################

def set_up_clenshaw_curtis(npts):
    # 4*npts + 1 CC quadrature
    wpts = 4*npts

    # M+1 point Clenshaw-Curtis quadrature
    def mesh(M):
        s = np.arange(0,M+1,1)
        ts = np.arange(1,M/2+1,1)
        ts = 2*ts - 1
        absc = np.cos(pi*s/M)
        wg = np.zeros(M+1)
        wg[0] = 1.0/(M**2-1.0)
        wg[-1] = wg[0]
        for ind in range(1,M):
            wg[ind] = 2*(-1)**ind/(M**2-1.0) + 4.0/M*np.sin(ind*pi/M)*np.sum(np.sin(ts*ind*pi/M)/ts)
        return absc,wg

    absc,wg = mesh(wpts)
    absc_2,wg_2 = mesh(int(wpts/2))
    wg_err = np.zeros(wpts+1)
    for itmp in range(int(wpts/2)+1):
        for jtmp in range(wpts+1):
            if abs(absc[jtmp] - absc_2[itmp]) < 1.e-20:
                wg_err[jtmp] = wg_2[itmp]
                break
    cinds = np.argsort(absc)
    np.savetxt('./grids/clenshaw_curtis_'+str(wpts+1)+'.csv',np.transpose((absc[cinds],wg[cinds],wg_err[cinds])),delimiter=',',header='Abscissa,Weight,Error weight',fmt='%.18f')
    return absc,wg,wg_err

################################################################################

# NIST DLMF subsection 3.5(iv)

def clenshaw_curtis_grid(npts):
    # 2*npts + 1 point CC quadrature

    def weights(M):
        k = np.arange(0,M+1,1)
        x = np.cos(pi*k/M)
        w = np.zeros(x.shape)
        if M%2 == 0:
            bfac = np.ones(floor(M/2.0))
            bfac[:-1]*=2
        else:
            bfac = 1.0

        for ik in range(M+1):
            sum =0.0
            j = np.arange(1,floor(M/2.0)+1,1)
            sum = np.sum(bfac*np.cos(2*j*k[ik]*pi/M)/(4*j**2-1.0))
            w[ik] = 1.0/M*(1.0 - sum)
            if 0 < k[ik] < M:
                w[ik] *= 2
        return x,w

    x,w = weights(2*npts)
    xerr,werr = weights(npts)
    wg_err = np.zeros(2*npts+1)
    wg_err[0::2] = werr
    cinds = np.argsort(x)
    np.savetxt('./grids/clenshaw_curtis_'+str(2*npts+1)+'.csv',np.transpose((x[cinds],w[cinds],wg_err[cinds])),delimiter=',',header='Abscissa,Weight,Error weight',fmt='%.18f')
    return x,w,wg_err

################################################################################

def qgauss(fun,bds,poly='legendre',npts=50,args=(),kwargs={}): # Very simple Gauss quadrature integrator, no error checking!
    if poly == 'legendre':
        default_grid = './grids/gauss_legendre_'+str(npts)+'_pts.csv'
    elif poly == 'laguerre':
        default_grid = './grids/gauss_laguerre_'+str(npts)+'_pts.csv'
    if poly == 'cheb':
        default_grid = './grids/gauss_cheb_'+str(npts)+'_pts.csv'
    if not path.isfile(default_grid) or path.getsize(default_grid)==0:
        gauss_quad(npts,grid_type=poly)
    wg,mesh = np.transpose(np.genfromtxt(default_grid,delimiter=',',skip_header=1))
    mesh = 0.5*(bds[1]-bds[0])*mesh + 0.5*(bds[1]+bds[0])
    wg *= 0.5*(bds[1]-bds[0])
    return np.sum(wg*fun(mesh,*args,**kwargs))

def cc(fun,bds,npts=500,args=(),kwargs={}): # Very simple Clenshaw-Curtis integrator, no error checking!
    default_grid = './grids/clenshaw_curtis_'+str(npts+1)+'.csv'
    if not path.isfile(default_grid) or path.getsize(default_grid)==0:
        clenshaw_curtis_grid(npts)
    mesh,wg,_ = np.transpose(np.genfromtxt(default_grid,delimiter=',',skip_header=1))
    mesh = 0.5*(bds[1]-bds[0])*mesh + 0.5*(bds[1]+bds[0])
    wg *= 0.5*(bds[1]-bds[0])
    return np.sum(wg*fun(mesh,*args,**kwargs))

def increase_cc_grid(fun,bds,of,ow,args=(),kwargs={}):

    nlen = 2*(len(of)-1)
    if nlen == 0:
        nlen = 1

    def grid(M):
        k = np.arange(0,M+1,1)
        x = np.cos(pi*k/M)
        w = np.zeros(x.shape)
        if M%2 == 0:
            bfac = np.ones(floor(M/2.0))
            bfac[:-1]*=2
        else:
            bfac = 1.0

        for ik in range(len(k)):
            sum =0.0
            j = np.arange(1,floor(M/2.0)+1,1)
            sum = np.sum(bfac*np.cos(2*j*k[ik]*pi/M)/(4*j**2-1.0))
            w[ik] = 1.0/M*(1.0 - sum)
            if 0 < k[ik] < M:
                w[ik] *= 2
        return 0.5*(bds[1]-bds[0])*x + 0.5*(bds[1]+bds[0]),0.5*(bds[1]-bds[0])*w

    tx,tw = grid(nlen)
    if nlen == 1:
        of = fun(tx,*args,**kwargs)
    else:
        tl = of
        of = np.zeros(nlen+1)#,dtype=tl.dtype
        of[0::2] = tl
        of[1::2] = fun(tx[1::2],*args,**kwargs)
    err_w = np.zeros(nlen+1)
    err_w[0::2] = ow
    return of,tw,err_w


def adap_cc(fun,bds,opt_d,args=(),kwargs={}):

    prec=1.e-6
    if 'prec' in opt_d:
        prec = opt_d['prec']

    min_depth = 2
    max_depth = 100
    osum = -1e20

    titer = 1
    nl = np.zeros(1)
    tw = np.zeros(1)
    for iter in range(max_depth):

        nl,tw,_=increase_cc_grid(fun,bds,nl,tw,args=args,kwargs=kwargs)
        tsum = np.sum(tw*nl)

        if abs(osum - tsum) < prec and iter >= min_depth:
            return tsum,{'code':1,'error':abs(osum - tsum)}
        else:
            osum = tsum
    if iter == max_depth-1:
        return tsum,{'code':0,'error':abs(osum - tsum)}

################################################################################

if __name__ == "__main__":

    def f(x):
        return x**2
    print(nquad(f,(0.0,2),'global_adap',{'itgr':'GK','prec':1.e-10,'npts':5}))
    #exit()

    def bpp(t):
        y = t#/(1.0 - t**2)
        dy = 1#(1.0 + t**2)/(1.0 - t**2)**2
        r = 1.0 - y*np.arctan(1.0/y)
        g = y**2*(3.0-y**2)/(1.0 + y**2)**3
        return np.log(r)*g*dy

    def bppp(t):
        y = t#2/(1.0 + t)-1.0
        dy = 1.0#2.0/(1.0 + t)**2
        r = 1.0 - y*np.arctan(1.0/y)
        return (2.0 + 1.0/(1.0 + y**2))*(1.0/(1.0 + y**2)**2/r)*dy

    bp2,bp2_err = nquad(bpp,(0.0,'inf'),'global_adap',{'itgr':'GK','prec':1.e-10,'npts':5,'err_meas':'quadpack'})
    bp3,bp3_err = nquad(bppp,(0.0,'inf'),'global_adap',{'itgr':'GK','prec':1.e-10,'npts':5,'err_meas':'quadpack'})

    bc = (5.0/9.0+2.0/np.pi*bp2+2.0/(9*np.pi)*bp3)/(3.0*pi**2)
    print(bc,bp2_err,bp3_err)
    exit()
    """
    exit()

    def h(y,pol):
        return (np.exp(-y**2)/(y-pol))
    for pol in np.arange(0.01,5.02,1.0):
        int,err=nquad(h,('-inf','inf'),'global_adap',{'itgr':'GK','prec':1.e-7},pars_ops={'PV':[pol]},args=(pol,))
        print(pol,int,err)
    #print(nquad(h,(-4.0,4.0),'global_adap',{'itgr':'GK','prec':1.e-7}))
    #print(nquad(h,(0.0,'inf'),'global_adap',{'itgr':'CC'}))
    exit()
    """
    def sig_x(x):
        f = 0.5 + (1.0 - x**2)/(4.0*x)*np.log(np.abs((1.0 + x)/(1.0 - x)))
        return x**2*f
    prec = 1.e-10
    print('Exact',0.25)
    print('Gauss-Legendre',qgauss(sig_x,(0.0,1.0)))
    print('Gauss-Chebyshev',qgauss(sig_x,(0.0,1.0),poly='cheb'))
    print('Double-exponential/Tanh-Sinh',dbl_exp(sig_x,(0.0,1.0),{'prec':prec}))
    print('Trapezoid',trap(sig_x,(1.e-6,1.0-1.e-6),{'prec':prec}))
    print('Simpson',trap(sig_x,(1.e-6,1.0-1.e-6),{'prec':prec,'simpson':True}))
    print('Global-adaptive Gauss-Kronrod',global_adap(sig_x,(0.0,1.0),{'itgr':'GK','prec':prec}))
    print('Global-adaptive Clenshaw-Curtis',global_adap(sig_x,(1.e-12,1.0-1.e-12),{'itgr':'CC','prec':prec}))
    print('Adaptive Clenshaw-Curtis',adap_cc(sig_x,(1.e-12,1.0-1.e-12),{'prec':prec}))
    exit()


    def f(x,fac=1.0):
        return np.exp(x*fac)
    def oof(x):
        return np.exp(-x)

    def g(x):
        return 4.0/(1.0 + x**2)

    #print(dbl_exp(g,(0.0,1.0),{}))
    print(nquad(oof,(0.0,'inf'),'global_adap',{'itgr':'GK','prec':1.e-10}))
    print(nquad(oof,(0.0,'inf'),'global_adap',{'itgr':'CC','prec':1.e-10}))
    #print(nquad(f,('-inf',0.0),'global_adap',{'itgr':'GK'}))
    #print(nquad(f,('-inf',0.0),'global_adap',{'itgr':'CC'}))
