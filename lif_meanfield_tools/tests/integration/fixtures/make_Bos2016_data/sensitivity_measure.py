import numpy as np
import matplotlib.pyplot as plt
import read_sim as rs 
import plot_helpers as ph
import h5py_wrapper.wrapper as h5
import meanfield.circuit as circuit

ph.set_fig_defaults()
circuit_params = ph.get_parameter_microcircuit()

def get_sensitivity_measure(calcAna, calcAnaAll, 
                            mode='gamma', eig_index=None):
    print 'Calculate sensitivity measure.'
    if calcAna or calcAnaAll:
        if mode == 'gamma' or mode == 'low':
            fmax = 100.0
        elif mode == 'high_gamma':
            fmax = 400.0
        circ = circuit.Circuit('microcircuit', circuit_params, 
                               fmax=fmax, from_file=not calcAnaAll)
        freqs, eigs = circ.create_eigenvalue_spectra('MH')
        if mode == 'gamma':
            fmax = freqs[np.argmin(abs(eigs[eig_index]-1))]
            Z = circ.get_sensitivity_measure(fmax)
            eigc = eigs[eig_index][np.argmin(abs(eigs[eig_index]-1))]
        elif mode == 'high_gamma':
            eigs = eigs[eig_index][np.where(freqs>150.)]
            freqs = freqs[np.where(freqs>150.)]
            fmax = freqs[np.argmin(abs(eigs-1.0))]
            fmax_index = np.argmin(abs(freqs-fmax))
            eigc = eigs[fmax_index]
            Z = circ.get_sensitivity_measure(fmax, index=eig_index)
        elif mode == 'low':
            eigs = eigs[eig_index]
            fmax = 0.0
            eigc = eigs[0]
            Z = circ.get_sensitivity_measure(fmax)
        k = np.asarray([1, 0])-np.asarray([eigc.real, eigc.imag])
        k /= np.sqrt(np.dot(k, k))
        k_per = np.asarray([-k[1], k[0]])
        k_per /= np.sqrt(np.dot(k_per, k_per))
        Z_amp = Z.real*k[0]+Z.imag*k[1]
        Z_freq = Z.real*k_per[0]+Z.imag*k_per[1]
        label = mode + str(eig_index)
        h5.add_to_h5('results.h5',{'sensitivity_measure':{label:{
            'f_peak': fmax, 'Z': Z, 'Z_amp': Z_amp, 'Z_freq': Z_freq, 
            'k': k, 'k_per': k_per, 'eigc': eigc}}},
            'a', overwrite_dataset=True)
    else:
        path_base = 'sensitivity_measure/' + mode + str(eig_index)
        fmax = h5.load_h5('results.h5', path_base + '/f_peak')
        Z = h5.load_h5('results.h5', path_base + '/Z')
        Z_amp = h5.load_h5('results.h5', path_base + '/Z_amp')
        Z_freq = h5.load_h5('results.h5', path_base + '/Z_freq')
        k = h5.load_h5('results.h5', path_base + '/k')
        k_per = h5.load_h5('results.h5', path_base + '/k_per')
        eigc = h5.load_h5('results.h5', path_base + '/eigc')
    return fmax, Z, Z_amp, Z_freq, k, k_per, eigc

def get_trajectories(calcAna, calcAnaAll):
    print 'Calculate eigenvalues.'
    if calcAna or calcAnaAll:
        circ = circuit.Circuit('microcircuit', circuit_params, 
                               from_file=not calcAnaAll)
        freqs, eigs = circ.create_eigenvalue_spectra('MH')
        eigs = np.transpose(eigs)
        h5.add_to_h5('results.h5',{'sensitivity_measure':{'freqs': freqs,
                     'eigs': eigs}}, 'a', overwrite_dataset=True)
    else:
        freqs = h5.load_h5('results.h5','sensitivity_measure/freqs')
        eigs = h5.load_h5('results.h5','sensitivity_measure/eigs')
    return freqs, eigs

def plot_matrices_gamma(M_title, Z, ax, fig, panel, fontsize_pops=7, fontsize_title=9):
    M_plot = [ph.reorder_matrix(Z[0]), ph.reorder_matrix(Z[1])]
    zmin = np.min(Z)
    zmax = np.max(Z)
    z = np.max([abs(zmin), abs(zmax)])
    for j,k in enumerate([3*panel, 3*panel+1]):
        im1 = ax[k].pcolor(M_plot[j], vmin=-z, vmax=z, cmap=plt.cm.coolwarm)
        ax[k].xaxis.tick_top()
        ax[k].set_xticks([0.5 + i for i in range(8)])
        ax[k].set_xticklabels(ph.populations, fontsize=fontsize_pops)
        ax[k].set_yticks([0.5 + i for i in range(8)])
        ax[k].set_xlabel(M_title[j])
        ax[k].set_title('sources\n', fontsize=fontsize_title)
    ax[0+3*panel].set_yticklabels(ph.populations[::-1], fontsize=fontsize_pops)
    ax[1+3*panel].set_yticklabels([])
    ax[0+3*panel].set_ylabel('targets', fontsize=fontsize_title)
    per = 0.9
    dMx = 0.07
    box = ax[0+3*panel].get_position()
    ax[0+3*panel].set_position([box.x0, box.y0, box.width * per, box.height])
    box = ax[1+3*panel].get_position()
    ax[1+3*panel].set_position([box.x0-box.width*(1-per)-dMx, box.y0, 
                                box.width * per, box.height])
    box = ax[1+3*panel].get_position()
    cbar_ax = fig.add_axes([box.x0 + box.width + 0.1*box.width, 
                            box.y0, 0.015, box.height])
    fig.colorbar(im1, cax=cbar_ax, ticks=[-0.8, -0.4, 0, 0.4, 0.8])

def plot_fig_gamma(calcAna, calcAnaAll, figname=None):
    plt.rcParams['figure.figsize'] = (6.929, 4.0)
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 7

    nx = 2
    ny = 3
    # index of dominant eigenmode
    eig_index = 4

    # initialise figure
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.33, hspace=0.4, top=0.91, 
                        bottom=0.07, left=0.07, right=0.99)
    ax = []
    # panel A
    ax.append(plt.subplot2grid((nx,ny), (0,0))) 
    ax.append(plt.subplot2grid((nx,ny), (0,1))) 
    # panel B
    ax.append(plt.subplot2grid((nx,ny), (0,2),rowspan=2)) 
    # panel C
    ax.append(plt.subplot2grid((nx,ny), (1,0))) 
    ax.append(plt.subplot2grid((nx,ny), (1,1)))

    # get data
    data = get_sensitivity_measure(calcAna, calcAnaAll, 'gamma', eig_index)
    fmax, Z, Z_amp, Z_freq, vec, vec_per, eigc = data

    ### panel A ###
    M_title = [r'$\Re(\mathbf{Z}(64\mathrm{Hz})$' + ')', 
               r'$\Im(\mathbf{Z}(64\mathrm{Hz})$' + ')']
    plot_matrices_gamma(M_title, [Z.real, Z.imag], ax, fig, 0)

    ### panel B ###
    # plot eigenvalue trajectory
    freqs, eigs = get_trajectories(calcAna, calcAnaAll)
    N = len(freqs)
    dc = 1/float(N)
    # loop over eigenvalues
    for i in range(N):
        ax[2].plot(eigs[i].real, eigs[i].imag, '.',
                   color=(0.9-0.9*i*dc, 0.9-0.9*i*dc, 0.9-0.9*i*dc),
                   markersize=1.0, zorder=1)
    # plot arrows
    sc = 0.04
    sce = 0.04
    head_width = 0.008
    width = 0.002
    # matrix elements of Z indicated by arrows
    conns = [[2,2],[3,3],[3,2],[2,3],
             [0,2],[0,3],[3,0],
             [1,1],[1,0],[0,1]]
    labels = [r'$4E\rightarrow 4E$', r'$4I\rightarrow 4I$', 
              r'$4E\rightarrow 4I$', r'$4I\rightarrow 4E$',
              r'$4E\rightarrow 23E$', r'$4I\rightarrow 23E$', 
              r'$23E\rightarrow 4I$', r'$23I\rightarrow 23I$', 
              r'$23E\rightarrow 23I$',r'$23I\rightarrow 23E$']
    colors = [(0.0,1.0,0.0),(0.0,0.8,0.0),(0.0,0.6,0.0),(0.0,0.4,0.0),
              (1.0,1.0,0.0),(0.8,0.8,0.0),(0.6,0.6,0.0),
              (1.0,0.0,0.0),(0.8,0.0,0.0),(0.6,0.0,0.0)]
    # critical eigenvalue
    ax[2].scatter(eigc.real, eigc.imag, s=20, color='black', zorder=2)
    # one in red
    ax[2].scatter(1,0, s=10, color='r')
    # arrows pointing towards one and in the perpendicular direction
    ax[2].arrow(eigc.real, eigc.imag, vec[0]*sc,vec[1]*sc, width=width, 
                head_width=head_width, color='black', zorder=2)
    ax[2].arrow(eigc.real, eigc.imag,vec_per[0]*sc, vec_per[1]*sc, 
                width=width, head_width=head_width, color='black', zorder=2)
    ax[2].annotate(r'$\mathbf{k}$', xy=(0.993, 0.018))
    ax[2].annotate(r'$\mathbf{k}_{\perp}$', xy=(0.988, 0.083))
    # arrows indicating elements of Z
    for i,c in enumerate(conns):
        ax[2].arrow(eigc.real, eigc.imag, Z[c[0]][c[1]].real*sce,
                    Z[c[0]][c[1]].imag*sce, width=width, 
                    head_width=head_width, color=colors[i], label=labels[i])
        ax[2].plot(-5, -5, color=colors[i], label=labels[i])
    ax[2].legend(loc='lower center', ncol=3)
    ax[2].set_xlim([0.9, 1.02])
    ax[2].set_ylim([-0.02, 0.1])
    ax[2].set_xlabel(r'$\Re(\lambda$)')
    ax[2].set_ylabel(r'$\Im(\lambda$)')
    ax[2].set_xticks([0.95, 1.0])
    ax[2].set_yticks([0.0, 0.05])
    box = ax[2].get_position()
    box4 = ax[4].get_position()
    per = 0.6
    ax[2].set_position([box.x0-0.02*box.height, box.y0+(1-per)*box.height, 
                        box.width+0.02*box.height, box.height * per])
    l = ax[2].bbox.transformed(ax[2].axes.transAxes.inverted())
    ax[2].legend(bbox_to_anchor=(-0.2, -0.7, 1.2, .95), loc='lower center', 
                 ncol=2, mode="expand", borderaxespad=0., fontsize=8.5)

    ### panel C ###
    M_title = [r'$\mathbf{Z}^{\mathrm{amp}}(64\mathrm{Hz})$',
               r'$\mathbf{Z}^{\mathrm{freq}}(64\mathrm{Hz})$']
    plot_matrices_gamma(M_title, [Z_amp, Z_freq], ax, fig, 1)

    for label in [['A',0],['B',2],['C',3]]:
        box = ax[label[1]].get_position()
        if label[0] == 'B':
              fig.text(box.x0-0.08,box.y0+box.height+0.02, label[0],
                       fontsize=13, fontweight='bold')
        else:
              fig.text(box.x0-0.03,box.y0+box.height+0.02, label[0],
                       fontsize=13, fontweight='bold')
    if figname is None:
        figname = 'sensitivity_measure_gamma.eps'
    plt.savefig(figname)

def plot_fig_high_gamma(calcAna, calcAnaAll, figname=None):
    plt.rcParams['figure.figsize'] = (6.929, 3)
    plt.rcParams['ytick.labelsize'] = 8.5
    plt.rcParams['xtick.labelsize'] = 8.5
    plt.rcParams['font.size'] = 9
    plt.rcParams['legend.fontsize'] = 6

    nx = 2
    ny = 4
    # indices of dominant eigenmodes
    eig_index = [0,1,2,3]
    # peak frequencies of dominant eigenmodes
    fps = [284, 267, 263, 241]

    # intialise figure
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.2, hspace=0.6, top=0.9, 
                        bottom=0.07, left=0.065, right=0.97)
    ax = []
    # panel A
    ax.append([plt.subplot2grid((nx,ny), (0,i)) for i in [0,1]])
    # panel B
    ax.append([plt.subplot2grid((nx,ny), (0,i)) for i in [2,3]]) 
    # panel C
    ax.append([plt.subplot2grid((nx,ny), (1,i)) for i in [0,1]]) 
    # panel D
    ax.append([plt.subplot2grid((nx,ny), (1,i)) for i in [2,3]])  

    # loop over panels
    for n in range(4):
        data = get_sensitivity_measure(calcAna, calcAnaAll, 
                            mode='high_gamma', eig_index=eig_index[n])
        fmax, Z, Z_amp, Z_freq, vec, vec_per, eigc = data
        M_plot = [ph.reorder_matrix(Z_amp), ph.reorder_matrix(Z_freq)]
        M_title = [r'$\mathbf{Z}^{\mathrm{amp}}($' + str(fps[n]) + r'$\mathrm{Hz})$',
                   r'$\mathbf{Z}^{\mathrm{freq}}($' + str(fps[n]) + r'$\mathrm{Hz})$']
        # normalise plots to dominant mode
        if n == 0:
            zmin = np.min([Z_amp, Z_freq])
            zmax = np.max([Z_amp, Z_freq])
            z = np.max([abs(zmin), abs(zmax)])
        tax = ax[n]
        for k in [1,0]:
            im1 = tax[k].pcolor(M_plot[k], vmin=-z, vmax=z, 
                                cmap=plt.cm.coolwarm)
            tax[k].xaxis.tick_top()
            tax[k].set_xticks([0.5 + i for i in range(8)])
            tax[k].set_xticklabels(ph.populations, fontsize=6)
            tax[k].set_yticks([0.5 + i for i in range(8)])
            tax[k].set_xlabel(M_title[k])
            tax[k].set_title('sources\n', fontsize=9)
        tax[0].set_yticklabels(ph.populations[::-1], fontsize=6)
        tax[1].set_yticklabels([])
        tax[0].set_ylabel('targets', fontsize=9)
        per = 0.9
        dMx = 0.02
        box = tax[0].get_position()
        tax[0].set_position([box.x0, box.y0, box.width * per, box.height])
        box = tax[1].get_position()
        tax[1].set_position([box.x0-box.width*(1-per)-dMx, box.y0, 
                             box.width * per, box.height])
        box = tax[1].get_position()
        if n%2 == 1:
            cbar_ax = fig.add_axes([box.x0 + box.width + 0.1*box.width, 
                                    box.y0, 0.015, box.height])
            fig.colorbar(im1, cax=cbar_ax, ticks=[-0.8, -0.4, 0, 0.4, 0.8])

    for i,label in enumerate(['A', 'B', 'C', 'D']):
        box = ax[i][0].get_position()
        fig.text(box.x0-0.04, box.y0+box.height+0.03, label[0],
                 fontsize=13, fontweight='bold')
    if figname is None:
        figname = 'sensitivity_measure_high_gamma.eps'
    plt.savefig(figname)

def plot_fig_low(calcAna, calcAnaAll, figname=None):
    plt.rcParams['figure.figsize'] = (3.346, 2.55)
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 6

    # intialise figure
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.2, hspace=0.6, top=0.85, 
                        bottom=0.09, left=0.18, right=0.83)
    ax = plt.subplot2grid((1,1), (0,0)) 

    # get data
    data = get_sensitivity_measure(calcAna, calcAnaAll, 
                                   mode='low', eig_index=6)
    fmax, Z, Z_amp, Z_freq, vec, vec_per, eigc = data
    # plot figure
    M_plot = [ph.reorder_matrix(Z_amp), ph.reorder_matrix(Z_freq)]
    M_title = [r'$\mathbf{Z}^{\mathrm{amp}}(0\mathrm{Hz})$',
               r'$\mathbf{Z}^{\mathrm{freq}}(0\mathrm{Hz})$']
    zmin = np.min([Z_amp, Z_freq])
    zmax = np.max([Z_amp, Z_freq])
    z = np.max([abs(zmin), abs(zmax)])
    im1 = ax.pcolor(M_plot[0], vmin=-z, vmax=z, cmap=plt.cm.coolwarm)
    ax.xaxis.tick_top()
    ax.set_xticks([0.5 + i for i in range(8)])
    tpops = ph.populations
    tpops[0] = '2/3E '
    ax.set_xticklabels(tpops, fontsize=9)
    ax.set_yticks([0.5 + i for i in range(8)])
    ax.set_xlabel(M_title[0])
    ax.set_title('sources\n', fontsize=12)
    ax.set_yticklabels(ph.populations[::-1], fontsize=9)
    ax.set_ylabel('targets', fontsize=12)
    per = 0.9
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * per, box.height])
    cbar_ax = fig.add_axes([box.x0 + box.width, box.y0, 0.04, box.height])
    fig.colorbar(im1, cax=cbar_ax, ticks=[-2.0, -1.0, 0, 1.0, 2.0])
    if figname is None:
        figname = 'sensitivity_measure_low.eps'
    plt.savefig(figname)
