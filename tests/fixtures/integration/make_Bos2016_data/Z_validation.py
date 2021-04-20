import numpy as np
import matplotlib.pyplot as plt
import read_sim as rs 
import plot_helpers as ph
import h5py_wrapper.wrapper as h5
import meanfield.circuit as circuit

ph.set_fig_defaults()
circuit_params = ph.get_parameter_microcircuit()

def get_spectra(calcAna, calcAnaAll, a, conn):
    print 'Calculate power spectra and eigenvalues.'
    label = str(a) + str(conn[0]) + str(conn[1])
    if calcAna or calcAnaAll:
        circ = circuit.Circuit('microcircuit', circuit_params, 
                    analysis_type='stationary', from_file=not calcAnaAll)
        I_new = circ.I.copy()
        Next_new = circ.Next.copy()
        I_new[conn[0]][conn[1]] = np.round(circ.I[conn[0]][conn[1]]*(1.0+a))
        # missing rate
        r = circ.th_rates[conn[1]]*circ.I[conn[0]][conn[1]]*abs(a) 
        # source inhibitory
        # a pos: more inh. input -> add ext. input 
        # a neg: less inh. input -> rem. ext. input
        if conn[1]%2 == 1: 
            Next_add = np.sign(a)*4*int(r/circ.v_ext)
        # source excitatory
        # a pos: more exc. input -> rem. ext. input
        # a neg: less exc. input -> add ext. input
        else:
            Next_add = -np.sign(a)*int(r/circ.v_ext)
        Next_new[conn[0]] = circ.Next[conn[0]] + Next_add
        circ.alter_default_params({'analysis_type': 'dynamical', 
                                   'I': I_new, 'Next': Next_new})
        freqs, power = circ.create_power_spectra()
        freqs, eigs = circ.create_eigenvalue_spectra('MH')
        rates = circ.th_rates
        h5.add_to_h5('results.h5',{'Z_validation':{label:{'freqs':freqs, 
                     'power':power, 'eigs': eigs, 'rates': rates}}},
                     'a', overwrite_dataset=True)
    else:
        freqs = h5.load_h5('results.h5','Z_validation/' + label + '/freqs')
        power = h5.load_h5('results.h5','Z_validation/' + label + '/power')
        eigs = h5.load_h5('results.h5','Z_validation/' + label + '/eigs')
        rates = h5.load_h5('results.h5','Z_validation/' + label + '/rates')
    return freqs, power, eigs, rates

def get_sim_data_spec(calcData, a, conn, pop):
    print 'Get power spectra from simulation.'
    label = str(a) + str(conn[0]) + str(conn[1])
    if conn == [3,3]:
        folder_base = 'data_zval_L4IL4I'
    elif conn == [5,4]:
        folder_base = 'data_zval_L5EL5I'
    folder = folder_base + str(1+a) 
    if calcData: 
        nr_window = 10
        bin_size = 1.0
        dt = (10000.0-300.0)/nr_window
        Ts = [300.0+i*dt for i in range(nr_window)]
        psim = 0
        for k,T in enumerate(Ts):
            fsim, p_sim = rs.get_spec(folder, pop, T+dt, T, 
                                      fmax=500.0, bin_size=bin_size)
            psim += np.asarray(p_sim)/float(len(Ts))
        powers = psim
        freqs = np.asarray(fsim)
        h5.add_to_h5('results.h5',{'Z_validation':{label:{
            'freqs_sim':freqs,'p_sim':powers}}},'a', overwrite_dataset=True)
    else:
        path_base = 'Z_validation/' + label
        freqs = h5.load_h5('results.h5', path_base + '/freqs_sim')
        powers = h5.load_h5('results.h5', path_base + '/p_sim')
    return freqs, powers

def get_sim_data_rate(calcData, a, conn, pop):
    print 'Get instantaneous firing rates from simulation.'
    label = str(a) + str(conn[0]) + str(conn[1])
    if conn == [3,3]:
        folder_base = 'data_zval_L4IL4I'
    elif conn == [5,4]:
        folder_base = 'data_zval_L5EL5I'
    folder = folder_base + str(1+a) 
    if True:#calcData: 
        bin_size = 15.
        dt = [2000,5000]
        time, rate = rs.get_inst_rate(folder, pop, dt[1], dt[0], 
                                      bin_size=bin_size)
        rate = rate[0].real
        h5.add_to_h5('results.h5',{'Z_validation':{label:{
            'time_sim': time,'inst_rate_sim': rate}}},'a', overwrite_dataset=True)
    else:
        path_base = 'Z_validation/' + label
        time = h5.load_h5('results.h5', path_base + '/time_sim')
        rate = h5.load_h5('results.h5', path_base + '/inst_rate_sim')
    return time, rate

def get_dotplot(calcData, a, conn, pop):
    print 'Get data for dot plot.'
    label = str(a) + str(conn[0]) + str(conn[1])
    if conn == [3,3]:
        folder_base = 'data_zval_L4IL4I'
    elif conn == [5,4]:
        folder_base = 'data_zval_L5EL5I'
    folder = folder_base + str(1+a)
    times = []
    gids = []
    dt = [1995,2030]
    if calcData:
        for pop in range(8):
            spikes, this_gids, gid_min, gid_max = rs.get_data_dotplot(
                folder, pop, dt[0],  dt[1])
            times.append(np.asarray(spikes)-dt[0])
            gids.append(abs(np.asarray(this_gids)-77169))
        h5.add_to_h5('results.h5',{'Z_validation':{label:{
            'times_dot_plot': times, 'gids': gids}}},'a', overwrite_dataset=True)
    else:
        path_base = 'Z_validation/' + label
        times  = h5.load_h5('results.h5', path_base + '/times_dot_plot')
        gids  = h5.load_h5('results.h5', path_base +'/gids')
    return times, gids

def plot_fig(calcData, calcAna, calcAnaAll, figname=None):
    plt.rcParams['figure.figsize'] = (7.5, 4.)
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['font.size'] = 11
    plt.rcParams['legend.fontsize'] = 7.8

    # maximum frequency for power spectrum plot
    fmax = 100.0
    # minimum and maximum frequency up to which eigenvalues will be shown
    fmin_eig = [50.0, 0.0]
    fmax_eig = [70.0, 10.0]
    # populations whose spectra are displayed
    pops = [2, 4]
    # indices of connections where the in-degrees are altered
    conns = [[3, 3] ,[5, 4]]
    # indices of eigenvalues corresponding to the dominant eigenmode
    eig_index = [4, 6]
    # factors by which in-degrees are altered
    alphas = [[0.0, 0.05, 0.1], [0.0, -0.15, -0.2]] 
    # shadings of gray for three alphas
    colors_spec_ana = [(0.0, 0.0, 0.0),(0.3, 0.3, 0.3),(0.6, 0.6, 0.6)]
    colors_spec_sim = [(0,0,1.0), (100/255., 150/255., 245/255.), 
                       (173/255., 216/255., 230/255.)]
    colors_array = [(0.0,0.0,0.45),(0.0,0.0,0.6),(0.0,0.0,0.85)]
    eig_label = [[r'$\alpha=$' + str(alphas[0][i]) 
                  for i in range(len(alphas[0]))],
                 [r'$\alpha=$' + str(alphas[1][i]) 
                  for i in range(len(alphas[1]))]]
    # size red dot
    sr = 15.
    # size black dot
    sb = 10.

    # gather data for spectra and eigenvalues in lists
    eigs = []
    powers = []
    p_sim = []
    rates_calc = []
    for i,conn in enumerate(conns):
        power = []
        eig = []
        power_sim = []
        time_sim = []
        rate_sim = []
        rate_calc = []
        time_dot_plot = []
        gid = []
        for j,a in enumerate(alphas[i]):
            freqs, p, teig, r_calc = get_spectra(calcAna, calcAnaAll, a, conn)
            power.append(p)
            eig.append(teig)
            rate_calc.append(r_calc)
            f_sim, p = get_sim_data_spec(calcData, a, conn, pops[i])
            power_sim.append(p)
        powers.append(power)
        eigs.append(eig)
        p_sim.append(power_sim)
        rates_calc.append(rate_calc)

    # gather data for dot plots and rates in lists
    times_sim = []
    rates_sim = []
    times_dot_plot = []
    gids = []
    for j,a in enumerate(alphas[0]):
        t, g = get_dotplot(calcData, a, conns[0], pops[0])
        times_dot_plot.append(t)
        gids.append(g)
    for j,a in enumerate(alphas[1]):
        t_sim, r = get_sim_data_rate(calcData, a, conns[1], pops[1])
        times_sim.append(t_sim)
        rates_sim.append(r)
        
    # intialise figure
    fig = plt.figure()
    fig.subplots_adjust(wspace=1.1, hspace=0.7, top=0.94, 
                        bottom=0.1, left=0.08, right=0.99)
    ax = [[],[]]
    for i,j in enumerate([0,3]):
        ax[i].append(plt.subplot2grid((6,6), (j,0), rowspan=3))
        ax[i].append(plt.subplot2grid((6,6), (j,1), colspan=2, rowspan=3))
        for k in [0,1,2]:
            ax[i].append(plt.subplot2grid((6,6), (j+k,3), colspan=3))

    ### plot eigenvalue trajectories ###
    # loop over panels
    for i in range(2):
        freqs_red = freqs[np.where(freqs<fmax_eig[i])]
        freqs_red = freqs_red[np.where(freqs_red>fmin_eig[i])]
        dc = 1/float(len(freqs_red))
        # loop over alteration factors
        for n,a in enumerate(alphas[i]):
            # loop over eigenvalues in trajectory corresponding to the
            # dominant mode
            k = 0
            for j,eig in enumerate(eigs[i][n][eig_index[i]]):
                if (freqs[j]>fmin_eig[i] and freqs[j]<fmax_eig[i]):
                    ax[i][0].plot(eig.real, eig.imag, '.',
                    color=(0.9-0.9*k*dc, 0.9-0.9*k*dc, 0.9-0.9*k*dc),
                    zorder=1)
                    k+=1
            # find eigenvalue closest to one
            emax_index = np.argmin(abs(eigs[i][n][eig_index[i]]-1.0))
            emax = eigs[i][n][eig_index[i]][emax_index]
            if n == 0:
                emax_index_original = emax_index
            emax_original = eigs[i][n][eig_index[i]][emax_index_original]
            ax[i][0].plot(emax.real, emax.imag, '+', mew=1, ms=6,
                color=colors_array[n], zorder=2, label=eig_label[i][n])
            ax[i][0].scatter(emax_original.real, emax_original.imag, s=sb,
                color=colors_spec_ana[n], zorder=2, label=eig_label[i][n])
        # plot one in red
        ax[i][0].scatter(1,0, s=sr, color='r', zorder=1)
        ax[i][0].set_ylabel(r'$\Im(\lambda(\omega)$)')
    ax[0][0].set_xlim([0.6, 1.05])
    ax[0][0].set_xticks([0.8, 1.0])
    ax[0][0].set_ylim([-0.05, 0.25])
    ax[0][0].set_yticks([0,0.1, 0.2])
    ax[1][0].set_xlim([0.0, 1.1])
    ax[1][0].set_xticks([0.5, 1.0])
    ax[1][0].set_ylim([-0.2, 0.025])
    ax[1][0].set_yticks([-0.1, 0])
    ax[1][0].set_xlabel(r'$\Re(\lambda(\omega)$)')

    ### plot spectra ###
    # loop over panels
    for k in range(2):
        j = pops[k]
        ymin = np.zeros(2)
        ymax = np.zeros(2)
        # loop over alteration factors
        # plot calculated spectra
        for i in range(len(alphas[k])):
            ax[k][1].plot(freqs, np.sqrt(powers[k][i][j]), 
                          color=colors_spec_ana[i], 
                          label=r'$\alpha$ = '+str(alphas[k][i]))
        # plot spectra from simulation
        for i in range(len(alphas[k])):
            ax[k][1].plot(f_sim, p_sim[k][i], color=colors_spec_sim[i])
        ax[k][1].set_yscale('log')
        ax[k][1].set_ylabel('power')
        ax[k][1].set_title(ph.populations[j])
        ax[k][1].set_xlim([4, fmax]) 
        ax[k][1].set_xticks([20, 40, 60, 80])
    ax[1][1].set_xlabel(r'frequency (1/$s$)')
    ax[0][1].set_ylim([1e-4, 1.2*1e-1])
    ax[0][1].set_yticks([1e-3, 1e-2])
    ax[1][1].set_ylim([5*1e-3,5*1e-1])
    ax[0][1].set_xticklabels([])
    ax[0][1].legend(loc='upper left')
    ax[1][1].legend(loc='upper left')

    ### plot dot plot ###
    for i in range(len(alphas[0])):
        ax[0][2+i].plot(times_dot_plot[i][2], gids[i][2], 'o', ms=1, mfc='0', mec='0')
        ax[0][2+i].tick_params(axis='y', colors=colors_spec_sim[i])
        ax[0][2+i].tick_params(axis='y', which=u'both',length=0)
        ax[0][2+i].set_ylim([np.min(gids[i][2]), np.max(gids[i][2])])
        ax[0][2+i].set_xticks([10,20,30])
        ax[0][2+i].set_yticks([np.mean(gids[i][2])])
        ax[0][2+i].set_yticklabels(['4E'])
        if i in [0,1]:
            ax[0][2+i].set_xticklabels([])

    ### plot instantaneous firing rate ###
    j = pops[1]
    for i in range(len(alphas[1])):
        ax[1][2+i].plot(times_sim[i], rates_sim[i], color=colors_spec_sim[i])
        ax[1][2+i].plot(times_sim[i], rates_calc[1][i][j]*np.ones_like(times_sim[i]), 
                        color=colors_spec_ana[i], linestyle='dashed')
        ax[1][2+i].set_ylim([4.5,12.5])
        ax[1][2+i].set_yticks([5,10])
        ax[1][2+i].set_xticks([3000,4000])
        if i in [0,1]:
            ax[1][2+i].set_xticklabels([])
        else:
            ax[1][2+i].set_xlabel(r't (ms)')
    ax[1][3].set_ylabel(r'$\bar{r}$ (1/s)')

    # resize sub-plots
    dx = 0.25
    for i in [0,1]:
        box = ax[i][2].get_position()
        new_height = box.height*(1+dx)
        dheight = new_height-box.height
        ax[i][2].set_position([box.x0, box.y0-dheight, 
                               box.width, new_height])
        box = ax[i][3].get_position()
        ax[i][3].set_position([box.x0, box.y0-dheight*0.5, 
                               box.width, new_height])
        box = ax[i][4].get_position()
        ax[i][4].set_position([box.x0, box.y0, 
                               box.width, new_height])
 
    for label in [['A', 0, 0, 0.08],['B', 1, 0, 0.08], 
                  ['C', 0, 2, 0.05], ['D', 1, 2, 0.05]]:
        box = ax[label[1]][label[2]].get_position()
        fig.text(box.x0-label[3], box.y0+box.height, label[0],
                 fontsize=13, fontweight='bold')
    if figname is None:
        figname = 'Z_validation.eps'
    plt.savefig(figname)
