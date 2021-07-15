import numpy as np

from constants import pi

"""
    NB: to use the FHNC data of
    M. Panholzer, M. Gatti, L. Reining, Phys. Rev. Lett. 120, 166402 (2018).
    DOI 10.1103/PhysRevLett.120.166402
    https://link.aps.org/doi/10.1103/PhysRevLett.120.166402

    please download it from:
    https://etsf.polytechnique.fr/research/connector/2p2h-kernel

    This code assumes the ``New kernel data with a 0.2 rs spacing'' file is used
    (~ 14 MB)
"""

# rs ranges from 0.8 bohr to 8.0 bohr in steps of 0.2 bohr
rs_min = 0.8
rs_max = 8.0
rs_step = 0.2
nrs = int((rs_max-rs_min)/rs_step) + 1
rs_l = np.linspace(rs_min,rs_max,nrs)

# q ranges from 0.1 kF to 6.4 kF in steps of 0.1 kF
q_min = 0.1
q_max = 6.4
q_step = 0.1
nq = int((q_max - q_min)/q_step) + 1
q_l = np.linspace(q_min,q_max,nq) # q

# omega ranges from 0 to 3.98 omega_p(0) in steps of 0.02 omega_p(0)
w_min = 0.0
w_max = 3.98
w_step = 0.02
nw = int((w_max - w_min)/w_step) + 1
w_l = np.linspace(w_min,w_max,nw) # omega

def get_fxc_single_rs(rs):

    rshun = int(100*rs)
    rshunstr = '{:}'.format(rshun)
    if rshun < 100:
        rshunstr = '0'+rshunstr

    kf = (9*pi/4)**(1/3)/rs
    ef = kf**2/2

    fxc_re = np.zeros((nq,nw))
    fxc_im = np.zeros((nq,nw))

    qblock = 0
    wblock = 0
    fl = './fxc_fhnc_tab/raw_data/fxc_'+rshunstr+'_2p2h_fine_L.txt'
    nheader = 7 # number of header lines

    with open(fl) as infl:
        for iln,ln in enumerate(infl):
            if iln <= nheader:
                continue
            wln = [float(tmp) for tmp in (ln.strip()).split()]
            if len(wln) == 0:
                qblock += 1
                wblock = 0
            else:
                # the data tables are in units of the bulk Fermi energy; we convert to atomic units
                fxc_im[qblock,wblock] = wln[8]/ef # Im f_xc(q,omega)
                fxc_re[qblock,wblock] = wln[9]/ef # Re f_xc(q,omega)
                wblock += 1
    return fxc_re,fxc_im

def write_fxc_to_file():

    from os import mkdir
    base_dir = './fxc_fhnc_tab/processed_data/'
    mkdir(base_dir)

    for irs,ars in enumerate(rs_l):

        ofl_name = base_dir + './fxc_2p2h_{:}.csv'.format(int(100*ars))
        re_fxc,im_fxc = get_fxc_single_rs(ars)
        ofl = open(ofl_name,'w+')
        ofl.write('rs={:.1f}, \n'.format(ars))
        ofl.write('Re fxc, Im fxc \n')
        for iq in range(nq):
            for iw in range(nw):
                ofl.write('{:.15e}, {:.15e} \n'.format(re_fxc[iq,iw],im_fxc[iq,iw]))
        ofl.close()

    return

def trilinear_interp(x,y,z,ix,iy,iz,x_tab,y_tab,z_tab,ftab):

    """
        assumes that ix corresponds to lower index, i.e.,
        x_tab[ix-1] <= x <= x_tab[ix]
        ditto for iy, iz
        reason: compatible with numpy.searchsorted
    """

    dxt = [ (x_tab[ix] - x)/(x_tab[ix]-x_tab[ix-1]), (x - x_tab[ix-1])/(x_tab[ix]-x_tab[ix-1]) ]
    dyt = [ (y_tab[iy] - y)/(y_tab[iy]-y_tab[iy-1]), (y - y_tab[iy-1])/(y_tab[iy]-y_tab[iy-1]) ]
    dzt = [ (z_tab[iz] - z)/(z_tab[iz]-z_tab[iz-1]), (z - z_tab[iz-1])/(z_tab[iz]-z_tab[iz-1]) ]

    fitp = 0.0 + 0.0j
    for jx in range(2):
        for jy in range(2):
            for jz in range(2):
                fitp += ftab[ix+jx-1, iy+jy-1, iz+jz-1]*dxt[jx]*dyt[jy]*dzt[jz]
    return fitp


def fxc_2p2h_lin_interp(rs,q,omega):

    from os.path import isfile

    mq = q.shape[0]
    mw = omega.shape[0]
    fxc_interp = np.zeros((mq,mw),dtype='complex')

    # rs should be scalar
    # q and omega can be vectors

    if (rs_l.min() > rs) or (rs_l.max() < rs):
        # out of scope!
        return fxc_interp

    irs = np.searchsorted(rs_l,rs)
    fl1 = './fxc_fhnc_tab/processed_data/fxc_2p2h_{:}.csv'.format(int(100*rs_l[irs-1]))
    fl2 = './fxc_fhnc_tab/processed_data/fxc_2p2h_{:}.csv'.format(int(100*rs_l[irs]))
    if not isfile(fl1) or not isfile(fl2):
        write_fxc_to_file()

    fxc_tmp = np.zeros((2,nq*nw),dtype='complex')
    fxc_tmp[0].real,fxc_tmp[0].imag = np.transpose(np.genfromtxt(fl1,delimiter=',',skip_header=2))
    fxc_tmp[1].real,fxc_tmp[1].imag = np.transpose(np.genfromtxt(fl2,delimiter=',',skip_header=2))

    fxc_tab = np.zeros((2,nq,nw),dtype='complex')
    for i in range(2):
        fxc_tab[i] = np.reshape(fxc_tmp[i],(nq,nw),order='C')

    kf = (9*pi/4)**(1/3)/rs
    q_ref = q_l*kf
    # only interpolate over those indices which are in scope
    qinds = np.arange(0,mq,1)[(q_ref.min() <= q) & (q <= q_ref.max())]
    iq_srt = np.searchsorted(q_ref,q)

    wp0 = (3/rs**3)**(0.5)
    w_ref = w_l*wp0
    winds = np.arange(0,mw,1)[(w_ref.min() <= omega) & (omega <= w_ref.max())]
    iw_srt = np.searchsorted(w_ref,omega)

    for jq in qinds:
        for jw in winds:
            fxc_interp[jq,jw] = trilinear_interp(rs,q[jq],omega[jw],1,iq_srt[jq],iw_srt[jw],[rs_l[irs-1],rs_l[irs]],q_ref,w_ref,fxc_tab)

    return fxc_interp

if __name__ == "__main__":

    rs_d = {'C': 1.3154461541085438, 'Al': 2.07, 'Si': 2.0087697565294302, 'Na': 3.93}
    conv = 27.211386245988 # hartree to eV
    for elt in rs_d:
        rs = rs_d[elt]
        kf = (9*pi/4)**(1/3)/rs
        wp0 = (3/rs**3)**(0.5)
        print(elt,' & ',round(rs,2),' & ',round(4*kf**2*conv,2),' & ',round(3.98*wp0*conv,2),' \\\\ ')
    exit()

    write_fxc_to_file()
