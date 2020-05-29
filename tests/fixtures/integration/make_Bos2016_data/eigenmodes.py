import numpy as np
import matplotlib.pyplot as plt
import plot_helpers as ph
import h5py_wrapper.wrapper as h5
import meanfield.circuit as circuit

ph.set_fig_defaults()
circuit_params = ph.get_parameter_microcircuit()


def get_eigenvalues(calcAna, calcAnaAll):
    print('Calculate eigenvalues.')
    if calcAna or calcAnaAll:
        circ = circuit.Circuit('microcircuit', circuit_params,
                               fmax=400.0, from_file=not calcAnaAll)
        freq_ev, eigenvalues = circ.create_eigenvalue_spectra('MH')
        h5.add_to_h5('results.h5', {'fig_eigenmodes': {'freq_ev': freq_ev,
                     'eigenvalues': eigenvalues}}, 'a', overwrite_dataset=True)
    else:
        freq_ev = h5.load_h5('results.h5', 'fig_eigenmodes/freq_ev')
        eigenvalues = h5.load_h5('results.h5', 'fig_eigenmodes/eigenvalues')
    return freq_ev, eigenvalues


def get_spectra(calcAna, calcAnaAll):
    print('Calculate spectra..')
    if calcAna or calcAnaAll:
        circ = circuit.Circuit('microcircuit', circuit_params,
                               fmax=400.0, from_file=not calcAnaAll)
        freqs, power = circ.create_power_spectra()
        h5.add_to_h5('results.h5', {'fig_eigenmodes': {'freqs': freqs,
                     'power': power}}, 'a', overwrite_dataset=True)
    else:
        freqs = h5.load_h5('results.h5', 'fig_eigenmodes/freqs')
        power = h5.load_h5('results.h5', 'fig_eigenmodes/power')
    return freqs, power


def get_spectra_approx(calcAna, calcAnaAll):
    print('Calculate approximate spectra.')
    if calcAna or calcAnaAll:
        circ = circuit.Circuit('microcircuit', circuit_params,
                               fmax=400.0, from_file=not calcAnaAll)
        freqs_approx, power_approx = circ.create_power_spectra_approx()
        h5.add_to_h5('results.h5', {'fig_eigenmodes': {
            'freqs_approx': freqs_approx, 'power_approx': power_approx}},
            'a', overwrite_dataset=True)
    else:
        freqs_approx = h5.load_h5('results.h5', 'fig_eigenmodes/freqs_approx')
        power_approx = h5.load_h5('results.h5', 'fig_eigenmodes/power_approx')
    return freqs_approx, power_approx


def plot_fig(calcAna, calcAnaAll):
    plt.rcParams['figure.figsize'] = (4.566, 3.346)
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['font.size'] = 9
    plt.rcParams['legend.fontsize'] = 7

    colors = ph.get_parameter_plot()
    ny = 3

    # initialise figure
    fig = plt.figure()
    fig.subplots_adjust(wspace=1.0, hspace=0.8, top=0.94,
                        bottom=0.12, left=0.12, right=0.97)
    ax = []
    ax.append(plt.subplot2grid((2, ny), (0, 0), rowspan=3))  # panel A
    ax.append(plt.subplot2grid((2, ny), (1, 0), colspan=2))  # panel B
    ax.append(plt.subplot2grid((2, ny), (1, 2)))  # panel C

    # panel B
    freq_ev, eigenvalues = get_eigenvalues(calcAna, calcAnaAll)
    power = [np.zeros(len(freq_ev), dtype=complex) for i in range(8)]
    labels = [r'$\lambda_0(\omega)$', r'$\lambda_1(\omega)$',
              r'$\lambda_2(\omega)$', r'$\lambda_3(\omega)$',
              r'$\lambda_4(\omega)$', r'$\lambda_5(\omega)$',
              r'$\lambda_6(\omega)$', r'$\lambda_7(\omega)$']
    for i in range(8):
        ax[1].plot(freq_ev, abs(1 / (1.0 - eigenvalues[i])), '.',
                   markersize=1.0, color=colors[i / 2][i % 2], label=labels[i])
    ax[1].set_xlim([10.0, 400.0])
    ax[1].set_ylim([2 * 1e-2, 3 * 1e1])
    ax[1].set_xticks([100, 200, 300])
    ax[1].set_yticks([1e-1, 1e0])
    ax[1].set_yscale('log')
    ax[1].set_xlabel('frequency $f$(1/$s$)')
    ax[1].set_ylabel('$|1/(1-\lambda)|$')
    ax[1].legend(loc='lower right', ncol=4, numpoints=1, labelspacing=0.2,
                 columnspacing=0.05, fontsize=5.5, markerscale=4.0)

    # panel C
    freqs, power = get_spectra(calcAna, calcAnaAll)
    freqs_approx, power_approx = get_spectra_approx(calcAna, calcAnaAll)
    ax[2].plot(freqs, np.sqrt(power[2]), color='gray')
    ax[2].plot(freqs, np.sqrt(power[0]), color='gray')
    ax[2].plot(freqs_approx, np.sqrt(power_approx[2]), color='black',
               linestyle='dashed')
    ax[2].plot(freqs_approx, np.sqrt(power_approx[0]), color='black',
               linestyle='dashed')
    ax[2].set_xlim([40.0, 90.0])
    ax[2].set_xticks([50, 70])
    ax[2].set_yscale('log')
    ax[2].set_yticks([1e-3, 1e-1])
    ax[2].set_ylim([1e-4, 2 * 1e-1])
    ax[2].set_xlabel('frequency $f$(1/$s$)')
    ax[2].set_ylabel('power')

    box = ax[0].get_position()
    for i, label in enumerate(['A', 'B', 'C']):
        box = ax[i].get_position()
        fig.text(box.x0 - 0.03, box.y0 + box.height + 0.01, label,
                 fontsize=13, fontweight='bold')
    plt.savefig('eigenmodes.eps')
