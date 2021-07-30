
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from constants import Eh_to_eV,clist

def plot_unlc(sign_conv=-1):

    metals = ['Al','Na']
    semiconds = ['Si','C']

    max_freq = [100,25] # in eV

    fig,ax = plt.subplots(1,figsize=(8,6))

    for isol,sol in enumerate(semiconds):

        flnm = './data_files/{:}/'.format(sol)+'alpha_omega_UNLC.csv'
        freq,alpha_re,alpha_im = np.transpose(np.genfromtxt(flnm,delimiter=',',skip_header=1))
        freq *= Eh_to_eV
        fmask = freq <= max_freq[0]
        freq = freq[fmask]
        alpha_re = sign_conv*alpha_re[fmask]
        alpha_im = sign_conv*alpha_im[fmask]

        ax.plot(freq,alpha_re,color=clist[isol],linestyle='-',linewidth=2)
        ax.plot(freq,alpha_im,color=clist[isol],linestyle='--',linewidth=2)

    ax.set_xlim([0,max_freq[0]])
    plt.show()


if __name__=="__main__":

    plot_unlc()
