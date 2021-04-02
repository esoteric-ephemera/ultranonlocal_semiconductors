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


def calc_alpha(fxcl,sph_avg=False):

    dyn_only_regex = ['DLDA','QV']

    print(('average density {:} bohr**(-3); rs_avg = {:} bohr').format(n_0_bar,(3.0/(4*pi*n_0_bar))**(1/3)))

    gmod,ng02,n_0_bar = init_n_g(crystal)
    n_avg2 = n_0_bar**2

    if not sph_avg:
        g_dot_q_hat = gmod**2

    Ng = gmod.shape[0]
    if bigrange:
        omega_l = np.linspace(0.0,400.0,500)/Eh_to_eV
    else:
        if crystal == 'Si':
            ulim = 25.0
        elif crystal == 'C':
            ulim = 25.0
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

        if sph_avg:
            ofl = './alpha_omega_sph_avg_'+fxc+addn+'.csv'
            # also only need to evaluate g_vec . q_hat once
            q_hat,intwg = init_ang_grid()
            Nq = q_hat.shape[0]
            g_dot_q_hat = np.zeros((Ng,Nq))
            for iag,ag in enumerate(g):
                g_dot_q_hat[iag] = (np.matmul(q_hat,ag))**2

            for iom,om in enumerate(omega_l):
                if fxc in dyn_only_regex:
                    fxc_g = wrap_kernel(0.0,om,n_0_bar,fxc)*np.ones(Ng)
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
                for iag in range(Ng):
                    alpha[iom] += np.sum(intwg*g_dot_q_hat[iag]*fxc_diff[iag]*ng02[iag])/n_avg2
        else:
            ofl = './alpha_omega_'+fxc+addn+'.csv'
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
                    if nproc == 1:
                        fxc_g = wrap_kernel(0.0,om,n_0_bar,fxc)*np.ones(Ng)
                        fxc_diff = fxc_g - fxc_g0
                    else:
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


        np.savetxt(ofl,np.transpose((omega_l,alpha.real,alpha.imag)),delimiter=',',header='omega (a.u.), Re alpha(w), Im alpha(w)',fmt='%.18f')

    return
