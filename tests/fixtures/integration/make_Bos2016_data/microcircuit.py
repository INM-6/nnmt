import numpy as np
import matplotlib.pyplot as plt
import read_sim as rs 
import plot_helpers as ph
import h5py_wrapper.wrapper as h5
import meanfield.circuit as circuit

T = 10000.0
folder = 'data_microcircuit'
ph.set_fig_defaults()
circuit_params = ph.get_parameter_microcircuit()

def get_rates(calcData, calcAna, calcAnaAll, folder=folder):
    print 'Get data for firing rates.'
    tmin = 300.0
    if calcData:
        rates_sim = []
        for pop in range(8):
            mu = rs.get_mu_rate(folder, pop, T, tmin)
            rates_sim.append(mu)
        h5.add_to_h5('results.h5',{'fig_microcircuit':{
            'rates_sim': rates_sim}}, 'a', overwrite_dataset=True)
    else:
        rates_sim = h5.load_h5('results.h5', 'fig_microcircuit/rates_sim')
    if calcAna or calcAnaAll:
        circ = circuit.Circuit('microcircuit', circuit_params, 
                               analysis_type='stationary', 
                               from_file=not calcAnaAll)
        rates_calc = circ.th_rates
        h5.add_to_h5('results.h5',{'fig_microcircuit':{
            'rates_calc': rates_calc}}, 'a', overwrite_dataset=True)
    else:
        rates_calc = h5.load_h5('results.h5','fig_microcircuit/rates_calc')
    return np.asarray(rates_sim), rates_calc

def get_dotplot(calcData, folder=folder):
    print 'Get data for dot plot.'
    times = []
    gids = []
    dt = [2000,2050]
    if calcData:
        for pop in range(8):
            spikes, this_gids, gid_min, gid_max = rs.get_data_dotplot(
                folder, pop, dt[0],  dt[1])
            times.append(np.asarray(spikes)-dt[0])
            gids.append(abs(np.asarray(this_gids)-77169))
        h5.add_to_h5('results.h5',{'fig_microcircuit':{
            'times': times, 'gids': gids}},'a', overwrite_dataset=True)
    else:
        times  = h5.load_h5('results.h5','fig_microcircuit/times')
        gids  = h5.load_h5('results.h5','fig_microcircuit/gids')
    return times, gids

def get_spec_sim(calcData, nr_window, bin_size):
    print 'Get power spectra from simulation.'
    power_sim = []
    if calcData:
        for pop in range(8):
            psim = 0
            dt = (10000.0-300.0)/nr_window
            Ts = [300.0+i*dt for i in range(nr_window)]
            for n,T in enumerate(Ts):
                fsim, p_sim = rs.get_spec(folder, pop, T+dt, T, 
                                          fmax=500.0, bin_size=bin_size)
                psim += np.asarray(p_sim)/float(len(Ts))
            power_sim.append(psim) 
            label = 'power' + str(pop) 
            h5.add_to_h5('results.h5',{'fig_microcircuit':{
                str(nr_window):{label: psim}}},'a',
                overwrite_dataset=True)
        freq_sim = np.asarray(fsim)
        h5.add_to_h5('results.h5',{'fig_microcircuit':{
            str(nr_window):{'freq_sim': fsim}}},'a',
            overwrite_dataset=True)
    else:
        for pop in range(8):
            label = 'power' + str(pop)
            power_sim.append(h5.load_h5('results.h5',
                    'fig_microcircuit/' + str(nr_window) + '/' + label))
        freq_sim = h5.load_h5('results.h5', 
                    'fig_microcircuit/' + str(nr_window) + '/freq_sim')
    return power_sim, freq_sim

def get_spec_ana(calcAna, calcAnaAll):
    print 'Calculate power spectra.'
    if calcAna or calcAnaAll:
        circ = circuit.Circuit('microcircuit', circuit_params, fmax=500.0, 
                               from_file=not calcAnaAll)
        freq_ana, power_ana = circ.create_power_spectra()
        h5.add_to_h5('results.h5',{'fig_microcircuit':{
            'freq_ana': freq_ana,'power_ana': power_ana}},
            'a',overwrite_dataset=True)
    else:
        freq_ana = h5.load_h5('results.h5','fig_microcircuit/freq_ana')
        power_ana = h5.load_h5('results.h5','fig_microcircuit/power_ana')
    return freq_ana, power_ana

def plot_fig(calcData=False, calcAna=False, calcAnaAll=False):
    plt.rcParams['figure.figsize'] = (6.929, 6.0)
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 10

    nr_windows = 20
    bin_size = 1.0

    colors = ph.get_parameter_plot()
    nx = 5
    ny = 4

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.93, 
                        bottom=0.185, left=0.1, right=0.97)
    ax = []
    ax.append(plt.subplot2grid((nx,ny), (0,0),rowspan=3)) # column
    ax.append(plt.subplot2grid((nx,ny), (0,3),rowspan=2)) # in-degrees
    ax.append(plt.subplot2grid((nx,ny), (2,3))) # firing rates
    ax.append(plt.subplot2grid((nx,ny), (0,1), rowspan=3, 
                               colspan=2)) # dotplot
    ax.append([plt.subplot2grid((nx,ny), (nx-(2-i%2),i/2)) 
               for i in range(8)]) # spectra  

    ### panel A ###
    ax[0].axis('off')

    ### panel B ###
    times, gids = get_dotplot(calcData)
    box = ax[3].get_position()
    ax[3].set_position([box.x0+box.width*0.05, box.y0, 
                        box.width*0.85, box.height])
    clrs=['0','0.5','0','0.5','0','0.5','0','0.5']
    pop_sizes = circuit_params['N']
    N_temp = 0
    for j in range(8):
        # pick a gid in the middle
        green_gids = gids[j][np.argmin(abs(abs(gids[j]-77169)-
                                           (N_temp+pop_sizes[j]/2.)))]
        green_times = times[j][np.where(gids[j]==green_gids)]
        if j == 2:
            k = 0
            while len(green_times) != 2:
                green_gid = gids[j][k]
                green_gids = [green_gid,green_gid]
                green_times = times[j][np.where(gids[j]==green_gid)]
                k += 1
                if k == len(gids[j]):
                    green_times = [green_times,green_times]
                    break
        N_temp += pop_sizes[j]
        ax[3].plot(times[j], gids[j], 'o', ms=1, mfc=clrs[j], mec=clrs[j])
        if len(green_times) >1:
            green_gids = [green_gids for i in range(len(green_times))]
        ax[3].plot(green_times, green_gids,'o',mfc='red',mec='red')
    # position labels
    size = 0
    layer_labels = []
    for layer in [3, 2, 1, 0]:
        layer_labels.append(sum(pop_sizes[2*layer:2*layer+2])/2. + size)
        size += sum(pop_sizes[2*layer:2*layer+2])
    ax[3].set_xlim(0,50)
    ax[3].set_ylim(0,77169)
    ax[3].set_xlabel(r't (ms)')
    ax[3].set_yticks(layer_labels)
    ax[3].set_yticklabels(['L2/3','L4','L5','L6'][::-1])
    ax[3].tick_params(axis='y',length=0)
    box = ax[3].get_position()
    ax[3].set_position([box.x0-box.width*0.07, box.y0, 
                        box.width, box.height])
    
    ### panel C ###
    In = circuit_params['I']
    M = ph.reorder_matrix(In)
    im1 = ax[1].pcolor(M, vmax=3000, cmap='OrRd')
    ax[1].xaxis.tick_top()
    ax[1].set_xticks([0.1, 1.45]+[0.5 + j for j in range(2,8)])
    ax[1].set_xticklabels(ph.populations, fontsize=6)
    ax[1].set_yticks([0.5 + j for j in range(8)])
    ax[1].set_yticklabels(ph.populations[::-1], fontsize=6)
    ax[1].tick_params(axis='x', which='both', top='off', labelbottom='off') 
    ax[1].tick_params(axis='y', which='both', right='off', left='off', 
                      labelbottom='off')
    ax[1].set_title('sources\n',fontsize=11)
    ax[1].set_ylabel('targets',fontsize=11)
    box = ax[1].get_position()
    per = 0.88
    cbar_ax = fig.add_axes([box.x0, box.y0*1.1, box.width, 0.015])
    ax[1].set_position([box.x0, box.y0*1.15, box.width, box.height*0.7])
    fig.colorbar(im1, orientation="horizontal", cax=cbar_ax, ticks=[0,3000]) 

    ### panel D ###
    rates_sim, rates_calc = get_rates(calcData, calcAna, calcAnaAll)
    ax[2].bar(np.arange(8), rates_sim*1000,0.8,
              color=['black','grey','black','grey',
                     'black','grey','black','grey'])
    ax[2].plot(np.arange(8)+0.4, rates_calc, 'x', mew=2.5, ms=5, color='red')
    ax[2].set_xticks([1, 3, 5, 7])
    ax[2].set_xticklabels(['L2/3', 'L4', 'L5', 'L6'])
    ax[2].set_yticks([2, 4, 6, 8])
    ax[2].set_yticks([1, 3, 5, 7])
    ax[2].set_ylabel(r'$\bar r$ (1/$s$)') 
    
    ### panel E ###
    power, freq = get_spec_sim(calcData, nr_windows, bin_size)
    power_1w, freq_1w = get_spec_sim(calcData, 1, bin_size)
    freq_ana, power_ana = get_spec_ana(calcAna, calcAnaAll)
    for layer in [0, 1, 2, 3]:
        for pop in [0, 1]:
            j = layer*2+pop
            box = ax[4][j].get_position()
            ax[4][j].set_position([box.x0, box.y0-box.height*0.58, 
                                   box.width, box.height])
            ax[4][j].plot(freq_1w, power_1w[j], color=(0.8, 0.8, 0.8))
            ax[4][j].plot(freq, power[j], color=(0.5, 0.5, 0.5))
            ax[4][j].plot(freq_ana,np.sqrt(power_ana[j]), 
                          color='black', zorder=2)
            ax[4][j].set_xlim([10.0, 400.0])
            ax[4][j].set_ylim([1e-6, 1e0])
            ax[4][j].set_yscale('log')
            ax[4][j].set_yticks([])
            ax[4][j].set_xticks([100, 200, 300])
            ax[4][j].set_title(ph.populations[j])
            if pop == 0:
                ax[4][j].set_xticklabels([])
            else:
                box = ax[4][j].get_position()
                ax[4][j].set_position([box.x0, box.y0-box.height*0.2, 
                                       box.width, box.height])
                ax[4][j].set_xlabel(r'frequency (1/$s$)')
    ax[4][0].set_yticks([1e-5,1e-3,1e-1])
    ax[4][1].set_yticks([1e-5,1e-3,1e-1])
    ax[4][0].set_ylabel(r'$|C(\omega)|$')
    ax[4][1].set_ylabel(r'$|C(\omega)|$')
    
    for i,label in enumerate(['A','C','D','B','E']):
        if label == 'E':
            box = ax[i][0].get_position()
        else:
            box = ax[i].get_position()
        fig.text(box.x0-0.04, box.y0+box.height+0.015, 
                 label,fontsize=16, fontweight='bold')
    plt.savefig('microcircuit.eps')
