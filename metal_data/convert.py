import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

pi = np.pi
clist=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:olive','tab:gray']
line_styles=['-','--','-.',':','-']
mkrlist=['o','s','d','^','v','x','*','+']

def convert_to_csv():

    to_process = ['./Al.dat','./Na.dat','./correlation-energy.dat']

    headers = ['','','']
    headers[0] = 'omega, Re MCP07, Im MCP07, Re MCP07 kbar=0, Im MCP07 kbar=0, Re Dynamic LDA, Im Dynamic LDA'
    headers[1] = headers[0]
    headers[2] = 'rs, PW92, RPA, ALDAxc, MCP07, MCP07 kbar=0, Dynamic LDA'

    for ifile,afile in enumerate(to_process):
        out_file = afile.split('.dat')[0]+'.csv'
        dat = []
        with open(afile,'r') as infl:
            for iln,aln in enumerate(infl):
                adat = [float(x) for x in aln.strip().split()]
                if len(adat)>0:
                    if afile == './Na.dat':
                        dat.append([adat[0],adat[3],adat[4],adat[1],adat[2],adat[5],adat[6]])
                    else:
                        dat.append(adat)
        np.savetxt(out_file,np.asarray(dat),delimiter=',',header=headers[ifile])

def metal_plots():

    olim = 10
    for sol in ['Al','Na']:
        fig,ax = plt.subplots(2,1,figsize=(8,6))
        max_bd = 0.0
        min_bd = 0.0

        alp_re = {}
        alp_im = {}
        flnm  = sol + '.csv'
        om,alp_re['DLDA'],alp_im['DLDA'],alp_re['MCP07_k0'],alp_im['MCP07_k0'],alp_re['MCP07'],alp_im['MCP07'] = np.transpose(np.genfromtxt(flnm,delimiter=',',skip_header=1))
        wind = np.argmin(np.abs(om - olim/2))
        for ifxc,fxc in enumerate(['MCP07','MCP07_k0','DLDA']):
            ax[0].plot(om,alp_re[fxc],color=clist[ifxc],linestyle=line_styles[ifxc])
            ax[1].plot(om,alp_im[fxc],color=clist[ifxc],linestyle=line_styles[ifxc])
            max_bd = max([max_bd,alp_re[fxc].max()])
            min_bd = min([min_bd,alp_im[fxc].min()])

        #ax[1].legend(fontsize=14)
        #ax[0].set_yticks(np.arange(0.0,max_bd,.1))
        #ax[1].set_yticks(np.arange(0.0,min_bd,-.1))
        ax[0].set_ylim([0.0,1.1*max_bd])#ax[0].get_ylim()[1]])
        ax[1].set_ylim([1.1*min_bd,0.0])
        ax[1].set_xlabel('$\\omega$ (eV)',fontsize=16)
        ax[0].set_ylabel('$\\mathrm{Re}~\\alpha(\\omega)$',fontsize=16)
        ax[1].set_ylabel('$\\mathrm{Im}~\\alpha(\\omega)$',fontsize=16)
        ax[0].yaxis.set_major_locator(MultipleLocator(.1))
        ax[0].yaxis.set_minor_locator(MultipleLocator(.05))
        ax[1].yaxis.set_major_locator(MultipleLocator(.05))
        ax[1].yaxis.set_minor_locator(MultipleLocator(.025))
        for i in range(2):
            ax[i].set_xlim([0.0,olim])#om.max()])
            ax[i].tick_params(axis='both',labelsize=14)
            ax[i].xaxis.set_major_locator(MultipleLocator(2))
            ax[i].xaxis.set_minor_locator(MultipleLocator(1))
        ax[0].xaxis.set_ticklabels([' ' for i in ax[0].xaxis.get_major_ticks()])
        ax[0].tick_params(axis='y',labelsize=14)
        plt.suptitle(sol,fontsize=16)
        for ifxc,fxc in enumerate(['MCP07','MCP07_k0','DLDA']):
            p2 = ax[0].transData.transform_point((om[wind+1],alp_re[fxc][wind+1]))
            p1 = ax[0].transData.transform_point((om[wind-1],alp_re[fxc][wind-1]))
            if fxc == 'DLDA':
                lbl = 'Dynamic LDA'
            elif fxc == 'MCP07_k0':
                lbl = 'MCP07, $\\bar{k}=0$'
            else:
                lbl = fxc
            if fxc == 'DLDA' and sol == 'Al':
                offset = -0.06
            else:
                offset = 0.01
            angle = 180/pi*np.arctan((p2[1]-p1[1])/(p2[0]-p1[0]))
            ax[0].annotate(lbl,(olim/2,alp_re[fxc][wind]+offset),color=clist[ifxc],fontsize=12,rotation=angle)
        plt.subplots_adjust(top=.93)
        #plt.show()
        #exit()
        plt.savefig('./'+sol+'_alpha_omega.pdf',dpi=600,bbox_inches='tight')
    return

def eps_c_plots():

    fig,ax = plt.subplots(figsize=(6,4))
    epsc = {}
    rs,epsc['PW92'],epsc['RPA'],epsc['ALDAxc'],_,epsc['MCP07'],epsc['MCP07_k0'],_ = np.transpose(np.genfromtxt('./correlation-energy.csv',delimiter=',',skip_header=1))
    ax.set_xlim([rs.min(),rs.max()])
    ax.set_ylim([-0.08,0.0])
    for ifnl,fnl in enumerate(epsc):
        ax.plot(rs,epsc[fnl],marker=mkrlist[ifnl],markersize=5,color=clist[ifnl],linestyle=line_styles[ifnl])
        if fnl in ['ALDAxc','RPA','PW92']:
            i = 5
            offset = .001
            p1 = ax.transData.transform_point((rs[i],epsc[fnl][i]))
            p2 = ax.transData.transform_point((rs[i+1],epsc[fnl][i+1]))
            angle = 180/pi*np.arctan((p2[1]-p1[1])/(p2[0]-p1[0]))
            ax.annotate(fnl,(0.5*(rs[i]+rs[i+1])-len(fnl)*0.1,0.5*(epsc[fnl][i]+epsc[fnl][i+1])+offset),rotation=angle,color=clist[ifnl],fontsize=12)
        elif fnl in ['MCP07','MCP07_k0']:
            if fnl == 'MCP07':
                i = 0
                txtpos = (2,-0.07)
                lbl = 'MCP07'
            else:
                i = 1
                txtpos = (3.6,-0.06)
                lbl = 'MCP07, $\\bar{k}=0$'
            ax.annotate(lbl,(0.5*(rs[i]+rs[i+1]),0.5*(epsc[fnl][i]+epsc[fnl][i+1])),xytext=txtpos,color=clist[ifnl],fontsize=12,arrowprops=dict(linewidth=1,color=clist[ifnl],arrowstyle='->'))
    ax.yaxis.set_major_locator(MultipleLocator(.01))
    ax.yaxis.set_minor_locator(MultipleLocator(.005))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel('$r_{\\mathrm{s}}$ (bohr)',fontsize=14)
    ax.set_ylabel('$\\varepsilon_{\\mathrm{c}}$ (hartree)',fontsize=14)
    ax.tick_params(axis='both',labelsize=12)
    #plt.show()
    plt.savefig('./ueg_epsilon_c.pdf',dpi=600,bbox_inches='tight')
    return

if __name__ == "__main__":

    #convert_to_csv()
    metal_plots()
    eps_c_plots()
