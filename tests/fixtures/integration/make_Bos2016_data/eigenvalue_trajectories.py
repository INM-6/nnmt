import numpy as np
import matplotlib.pyplot as plt
import read_sim as rs 
import plot_helpers as ph
import h5py_wrapper.wrapper as h5
import meanfield.circuit as circuit

ph.set_fig_defaults()
circuit_params = ph.get_parameter_microcircuit()

def get_eigenvalues(calcAna, calcAnaAll):
    print 'Calculate eigenvalues.'
    if calcAna or calcAnaAll:
        circ = circuit.Circuit('microcircuit', circuit_params, 
                               fmax=400.0, from_file=not calcAnaAll)
        freqs, eigs = circ.create_eigenvalue_spectra('MH')
        eigs = np.transpose(eigs)
        h5.add_to_h5('results.h5',{'eigenvalue_trajectories':{
            'freqs': freqs, 'eigs': eigs}}, 
            'a', overwrite_dataset=True)
    else:
        freqs = h5.load_h5('results.h5','eigenvalue_trajectories/freqs')
        eigs = h5.load_h5('results.h5', 'eigenvalue_trajectories/eigs')
    return freqs, eigs

def get_eigenvalues_layers(calcAna, calcAnaAll):
    print 'Calculate eigenvalues.'
    if calcAna or calcAnaAll:
        circ = circuit.Circuit('microcircuit', circuit_params, 
                               fmax=400.0, from_file=not calcAnaAll)
        eigs_layer = []
        for i,layer in enumerate(ph.layers):
            M_red = np.zeros((8,8))
            M_red[2*i:2*i+2, 2*i:2*i+2] = np.ones((2,2))
            circ.reduce_connectivity(M_red)
            freqs, eigs = circ.create_eigenvalue_spectra('MH')
            eigs_layer.append(eigs[0])
            eigs_layer.append(eigs[1])
            circ.restore_full_connectivity()
        eigs_layer = np.transpose(np.asarray(eigs_layer))
        h5.add_to_h5('results.h5',{'eigenvalue_trajectories':{
            'freq_layer': freqs, 'eigs_layer': eigs_layer}},
            'a',overwrite_dataset=True)
    else:
        freqs = h5.load_h5('results.h5','eigenvalue_trajectories/freq_layer')
        eigs_layer = h5.load_h5('results.h5',
                                'eigenvalue_trajectories/eigs_layer')
    return freqs, eigs_layer

def get_spectra(calcAna, calcAnaAll):
    print 'Calculate spectra.'
    if calcAna or calcAnaAll:
        circ = circuit.Circuit('microcircuit', circuit_params, 
                               fmax=400.0, from_file=not calcAnaAll)
        freqs, power = circ.create_power_spectra()
        power_layer = []
        for i,layer in enumerate(ph.layers):
            M_red = np.zeros((8,8))
            M_red[2*i:2*i+2, 2*i:2*i+2] = np.ones((2,2))
            circ.reduce_connectivity(M_red)
            freqs, power_red = circ.create_power_spectra()
            power_layer.append([power_red[2*i],power_red[2*i+1]])
            circ.restore_full_connectivity()
        h5.add_to_h5('results.h5',{'eigenvalue_trajectories':{
            'freqs_spec': freqs, 'power_layer': power_layer, 
            'power': power}},'a',overwrite_dataset=True)
    else:
        freqs = h5.load_h5('results.h5','eigenvalue_trajectories/freqs_spec')
        power_layer = h5.load_h5('results.h5',
                                 'eigenvalue_trajectories/power_layer')
        power = h5.load_h5('results.h5','eigenvalue_trajectories/power')
    return freqs, power_layer, power

def plot_fig(calcAna, calcAnaAll, figname=None):
    plt.rcParams['figure.figsize'] = (6.929, 3.5)
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['font.size'] = 9
    plt.rcParams['legend.fontsize'] = 6

    colors = ph.get_parameter_plot()
    colors_array = [(0.0,0.0,0.45),(0.0,0.0,0.6),(0.0,0.0,0.85),
                    (0.0,0.0,1.0),(0.4,0.4,0.4),'black']
    
    def get_color(i, l):
        cont_colors = [(1.0-i*dc, 0.0, 0.0), (0.0, 1.0-i*dc, 0.0), 
                       (1.0-i*dc, 0.0, 1.0-i*dc), (0.0, 0.0, 1.0-i*dc)]
        index = ph.layers.index(l)
        return cont_colors[index]

    # initialise figure
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.5, hspace=0.8, top=0.94, 
                        bottom=0.11, left=0.09, right=0.99)
    ax = []
    # panel A
    ax.append(plt.subplot2grid((4,3), (0,0),rowspan=2)) 
    ax.append(plt.subplot2grid((4,3), (2,0)))
    # panel B
    ax.append(plt.subplot2grid((4,3), (0,1),sharey=ax[0], 
                               sharex=ax[0], rowspan=2)) 
    ax.append(plt.subplot2grid((4,3), (2,1),sharey=ax[1],sharex=ax[1]))
    # panel C
    ax.append(plt.subplot2grid((4,3), (0,2),rowspan=2)) 
    ax.append(plt.subplot2grid((4,3), (2,2),rowspan=2))

    ### panel A ###
    ## top ##
    freqs, eigs = get_eigenvalues(calcAna, calcAnaAll)
    N = len(freqs)
    dc = 1/float(N)
    # plot every third eigenvalue
    for i in range(0, N, 3):
        ax[0].plot(eigs[i].real, eigs[i].imag, '.',
                   color=(0.9-0.9*i*dc, 0.9-0.9*i*dc, 0.9-0.9*i*dc),
                   markersize=1.0, zorder=1)
    ax[0].scatter(1,0, s=15, color='r')
    ax[0].set_ylim([-4, 6.5])
    ax[0].set_xlim([-11.5, 2])
    ax[0].set_xticks([-9, -6, -3, 0])
    ax[0].set_yticks([-3, 0, 3, 6])
    ax[0].set_ylabel('Im($\lambda(\omega)$)')

    ## bottom ##
    eigst = np.transpose(eigs)
    eigcs = []
    for j in [0,1,2,3,4,6]:
        fmax = freqs[np.argmin(abs(eigst[j]-1.0))]
        fmax_index = np.argmin(abs(freqs-fmax))
        eigc = eigst[j][fmax_index]
        eigcs.append(eigc)

    N = len(freqs)
    dc = 1/float(N)
    # plot every third eigenvalue
    for i in range(0, N, 3):
        ax[1].plot(eigs[i].real, eigs[i].imag, '.',
                   color=(0.9-0.9*i*dc, 0.9-0.9*i*dc, 0.9-0.9*i*dc),
                   markersize=1.0)
    # frequencies where eigenvalue trajectory is closest to one
    fmaxs = [284,268,263,237,63,0]
    markers = ['<', '>', '^','v', 'o','+']
    for i in range(6):
        eigc = eigcs[i]
        ax[1].plot(eigc.real, eigc.imag, markers[i], color='black',#colors_array[i],
                   mew=1, ms=4, label=str(fmaxs[i])+'Hz')
    ax[1].legend(bbox_to_anchor=(-0.35, -0.9, 1.6, 0.5), loc='center', 
                 ncol=3, mode="expand", borderaxespad=3.5, fontsize=7,
                 numpoints = 1)
    ax[1].scatter(1,0, s=5, color='r')
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0-box.height, box.width, box.height*2])
    ax[1].set_xlabel('Re($\lambda(\omega)$)')
    ax[1].set_ylabel('Im($\lambda(\omega)$)')
    ax[1].set_ylim([-0.3, 0.5])
    ax[1].set_xlim([0.1, 1.1])
    ax[1].set_yticks([-0.2, 0, 0.2, 0.4])
    ax[1].set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])

    ### panel B ###
    ## top ##
    freqs, eigs = get_eigenvalues_layers(calcAna, calcAnaAll)
    N = len(freqs)
    dc = 1/float(N)
    for k,l in enumerate(ph.layers):
        # plot every third eigenvalue
        for i in range(0, N, 3):
            j = k*2 + i
            if i == 0:
                ax[2].plot(eigs[i,2*k:2*k+2].real, eigs[i,2*k:2*k+2].imag,
                           '.', color=get_color(i, l), label=l,
                           markersize=1.0)
            else:
                ax[2].plot(eigs[i,2*k:2*k+2].real, eigs[i,2*k:2*k+2].imag,
                           '.', color=get_color(i, l), markersize=1.0)
    ax[2].scatter(1,0, s=15, color='r')
    ax[2].legend(loc='lower right', ncol=2)
    ax[2].set_xlim([-11.5, 2])

   ## bottom ##
    eigst = np.transpose(eigs)
    eigcs = []
    for j in [3,7,5,1,0,4]:
        fmax = freqs[np.argmin(abs(eigst[j]-1.0))]
        fmax_index = np.argmin(abs(freqs-fmax))
        eigc = eigst[j][fmax_index]
        eigcs.append(eigc)

    N = len(freqs)
    dc = 1/float(N)
    for k,l in enumerate(ph.layers):
        # plot every third eigenvalue
        for i in range(0, N, 3):
            ax[3].plot(eigs[i,2*k:2*k+2].real, eigs[i,2*k:2*k+2].imag,
                       '.', color=get_color(i, l), markersize=1.0)
    # frequencies where eigenvalue trajectory is closest to one
    fmaxs = [284,267,263,241,84,0]
    markers = ['<', '>', '^','v', 'o','+']
    for i in range(6):
        eigc = eigcs[i]
        ax[3].plot(eigc.real, eigc.imag, markers[i], color='black',#colors_array[i],
                   mew=1, ms=4, label=str(fmaxs[i])+'Hz')
    ax[3].legend(bbox_to_anchor=(-0.35, -0.9, 1.6, 0.5), loc='center', 
                 ncol=3, mode="expand", borderaxespad=3.5, fontsize=7, 
                 numpoints = 1)
    ax[3].scatter(1, 0, s=5, color='r')
    box = ax[3].get_position()
    ax[3].set_position([box.x0, box.y0-box.height, box.width, box.height*2])
    ax[3].set_xlabel('Re($\lambda(\omega)$)')

    ### panel C ##
    freqs, power_layer, power = get_spectra(calcAna, calcAnaAll)
    for i in [0,1]:
        for j in [0,1]:
            ax[4+i].plot(freqs, np.sqrt(np.asarray(power_layer[i][j])), 
                         color=colors[i][j])
            ax[4+i].plot(freqs, np.sqrt(np.asarray(power[2*i+j])), 
                         color='black', linestyle='dashed')
    for i in [4,5]:
        ax[i].set_yscale('log')
        ax[i].set_xticks([100, 200, 300])
        ax[i].set_ylabel('power')
        ax[i].set_yticks([1e-2, 1e-4])
    ax[4].set_ylim([5*1e-6, 5*1e-2])
    ax[5].set_ylim([2*1e-5, 4*1e-1])
    ax[5].set_xlabel('frequency $f$(1/$s$)')

    for i,label in enumerate(['A','B','C']):
        box = ax[2*i].get_position()
        fig.text(box.x0-0.03, box.y0+box.height+0.01, label,
                 fontsize=13, fontweight='bold')
    if figname is None:
        figname = 'eigenvalue_trajectories.eps'
    plt.savefig(figname)
