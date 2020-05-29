import numpy as np
import matplotlib.pyplot as plt
import read_sim as rs 
import plot_helpers as ph
import h5py_wrapper.wrapper as h5
import meanfield.circuit as circuit

ph.set_fig_defaults()
circuit_params = ph.get_parameter_microcircuit()

def get_spectra(panel, calcAna, calcAnaAll):
    print 'Calculate power spectra.'
    eig_index = 4
    if calcAna or calcAnaAll:
        circ = circuit.Circuit('microcircuit', circuit_params, 
                               fmax=100.0, from_file=not calcAnaAll)
        freqs, power = circ.create_power_spectra()
        h5.add_to_h5('results.h5',{'oscillation_origin':{panel:{
            'freqs': freqs, 'power': power}}},'a',overwrite_dataset=True)
        if panel == 'A':
            Mred = np.zeros((8,8))
            Mred[0,0:4] = 1
            Mred[1,0:2] = 1
            Mred[2:4,2:4] = 1
            Mred[3,0] = 1
        elif panel == 'B':
            Mred = np.ones((8,8))
            Mred[3,0] = 0
        circ.reduce_connectivity(Mred)
        freqs_sub, power_sub = circ.create_power_spectra()
        h5.add_to_h5('results.h5',{'oscillation_origin':{panel:{
            'power_sub': power_sub, 'Mred': Mred}}},
            'a',overwrite_dataset=True)
    else:
        path_base = 'oscillation_origin/' + panel
        freqs = h5.load_h5('results.h5', path_base + '/freqs')
        power = h5.load_h5('results.h5', path_base + '/power')
        power_sub = h5.load_h5('results.h5', path_base + '/power_sub')
        Mred = h5.load_h5('results.h5', path_base + '/Mred')
    return freqs, power, power_sub, Mred

def get_spectra_sim_av(folder, pop, nr_window, bin_size):
    psim = 0
    dt = (10000.0-300.0)/nr_window
    Ts = [300.0+i*dt for i in range(nr_window)]
    for n,T in enumerate(Ts):
        fsim, p_sim = rs.get_spec(folder, pop,T=T+dt, tmin=T, 
                                  fmax=500.0, bin_size=bin_size)
        psim += np.asarray(p_sim)/float(len(Ts))
    return psim, fsim

def get_spectra_sim(panel, calcData):
    print 'Get power spectra from simulation.'
    folder = 'data_iso_panel' + panel
    T = 10000.0
    tmin = 300.0
    nr_windows = 20
    bin_size = 1.0
    power_sim = [] 
    power_sim_av = []
    if calcData:
      for pop in range(4):
          freq_sim, p_sim = rs.get_spec(folder, pop, T, tmin, 
                                        fmax=100.0, bin_size=bin_size)
          power_sim.append(p_sim)
          label = 'power' + str(pop) + '_sim'
          h5.add_to_h5('results.h5',{'oscillation_origin':{panel:{
              label: p_sim}}}, 'a', overwrite_dataset=True)
          psim, fsim = get_spectra_sim_av(folder, pop, nr_windows, bin_size)
          freq_sim_av = fsim
          power_sim_av.append(psim)
          h5.add_to_h5('results.h5',{'oscillation_origin':{panel:{
              label + '_av': psim}}},'a',overwrite_dataset=True)
      h5.add_to_h5('results.h5',{'oscillation_origin':{panel:{
                   'freq_sim': freq_sim, 'freq_sim_av': fsim}}}, 
                   'a', overwrite_dataset=True)
    else:
        path_base = 'oscillation_origin/' + panel
        freq_sim = h5.load_h5('results.h5', path_base + '/freq_sim')
        freq_sim_av = h5.load_h5('results.h5', path_base + '/freq_sim_av')
        for pop in range(4):
            label = '/power' + str(pop) + '_sim'
            power_sim.append(h5.load_h5('results.h5', path_base + label))
            power_sim_av.append(h5.load_h5('results.h5', 
                                           path_base + label + '_av'))
    return freq_sim, power_sim, freq_sim_av, power_sim_av

def plot_connectivity_matrix(Mred, ax):
    M = ph.reorder_matrix(Mred)
    im1 = ax.pcolor(M, cmap='Greys')
    ax.xaxis.tick_top()
    ax.set_xticks([0.5 + i for i in range(8)])
    ax.set_xticklabels(ph.populations, fontsize=7)
    ax.set_yticks([0.5 + i for i in range(8)])
    ax.set_yticklabels(ph.populations[::-1], fontsize=7)
    ax.set_title('sources\n', fontsize=10)
    ax.set_ylabel('targets', fontsize=10)

def plot_spectra(panel, data_sim, data_calc, ax):
    freq_sim, power_sim, freq_sim_av, power_sim_av = data_sim
    freqs, power, power_sub, Mred = data_calc
    ymin = np.zeros(4)
    ymax = np.zeros(4)
    boxes = []
    for i,k in enumerate([0,2,3]):
        axt = ax[i+1+panel*4] 
        axt.plot(freq_sim, power_sim[k], color=(0.8, 0.8, 0.8))
        axt.plot(freq_sim_av, power_sim_av[k], color=(0.5, 0.5, 0.5))
        axt.plot(freqs, np.sqrt(np.asarray(power_sub[k])), 'black')
        axt.plot(freqs, np.sqrt(np.asarray(power[k])), 
                 color='black', linestyle='dashed')
        axt.set_yscale('log')
        axt.set_xticks([20, 40, 60, 80])
        axt.set_xlim([0, 100])
        axt.set_yticks([1e-5, 1e-3, 1e-1])
        ymin[k], ymax[k] = axt.get_ylim()
        axt.set_title(ph.populations[k])
        if k>0:
            axt.set_yticklabels([])
        box = axt.get_position()
        x_spec = 0.35
        if i == 0:
              w = box.width
              axt.set_position([box.x0+w*x_spec, box.y0, 
                                box.width-w*x_spec*0.3, box.height])
        else:
              axt.set_position([boxes[-1].x0+boxes[-1].width+0.026, 
                                box.y0, box.width-w*x_spec*0.3, box.height])
        boxes.append(axt.get_position())
    ax[1+panel*4].set_ylabel('power') 
    for i in range(3):
        ax[i+1+panel*4].set_ylim([1e-6, 2*5e-1])

def plot_fig(calcData, calcAna, calcAnaAll, figname=None):
    plt.rcParams['figure.figsize'] = (6.929, 4.0)
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 7

    # initialise figure
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.91, 
                        bottom=0.11, left=0.07, right=0.97)
    # panel A
    ax = [plt.subplot2grid((2, 12), (0, i), colspan=3) 
          for i in [0, 3, 6, 9]]
    # panel B
    ax += [plt.subplot2grid((2, 12), (1, i), colspan=3) 
           for i in [0, 3, 6, 9]]
     
    ### panel A ###
    data_calc = get_spectra('A', calcAna, calcAnaAll)
    data_sim = get_spectra_sim('A', calcData)
    # connectivity matrix
    plot_connectivity_matrix(data_calc[3], ax[0])
    # spectra
    plot_spectra(0, data_sim, data_calc, ax)

    ### panel B ###
    data_calc = get_spectra('B', calcAna, calcAnaAll)
    data_sim = get_spectra_sim('B', calcData)
    # connectivity matrix
    plot_connectivity_matrix(data_calc[3], ax[4])
    # spectra
    plot_spectra(1, data_sim, data_calc, ax)
    ax[6].set_xlabel('frequency $f$(1/$s$)')

    for label in [['A',0],['B',4]]:
        box = ax[label[1]].get_position()
        fig.text(box.x0-0.04, box.y0+box.height+0.03, 
                 label[0],fontsize=13, fontweight='bold')
    if figname is None:
        figname = 'oscillation_origin.eps'
    plt.savefig(figname)



