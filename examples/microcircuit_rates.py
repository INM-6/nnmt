"""
Microcircuit Firing Rates
=========================

Here we calculate the firing rates of the :cite:t:`potjans2014` microcircuit
model.
"""

import nnmt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import svgutils for inserting sketch if available
try:
    import os
    import svgutils.transform as sg
    insert_sketch = True
except ImportError:
    insert_sketch = False

# use matplotlib style file
plt.style.use('frontiers.mplstyle')

###############################################################################
# First we create a network model of the microcircuit, passing the parameter
# yaml file.
microcircuit = nnmt.models.Microcircuit(
    '../tests/fixtures/integration/config/Bos2016_network_params.yaml')

###############################################################################
# Then we simply calculate the firing rates for exponentially shape post
# synaptic currents, by calling the respective function and passing the
# microcircuit. Here we chose to use the 'taylor' method for calculating the
# firing rates.
firing_rates = nnmt.lif.exp.firing_rates(microcircuit, method='shift')

print(f'Mean rates: {firing_rates}')

###############################################################################
# Then we compare the rates to the publicated data from :cite:t:`bos2016`. We
# load the simulated rates using the data stored as integration test fixtures.
# Note that the original data use rates in 1/ms, which we need to convert to
# Hz.
fix_path = '../tests/fixtures/integration/data/'
result = nnmt.input_output.load_h5(
    fix_path + 'Bos2016_publicated_and_converted_data.h5')
simulated_rates = result['fig_microcircuit']['rates_sim'] * 1000

print(f'Mean simulated rates: {simulated_rates}')

###############################################################################
# Finally, we plot the rates together in one plot.
width = 0.03937007874 * 80
height = width * 1.3

fig = plt.figure(figsize=(width, height),
                 tight_layout=True)

sim_colors = ['#4c72b0', '#c44e52']
thy_color = '#ff8f2fff'

ax0 = plt.subplot2grid((5, 1), (0, 0), rowspan=3, colspan=1)
ax0.set_axis_off()
ax1 = plt.subplot2grid((5, 1), (3, 0), rowspan=2, colspan=1)

# add labels to panels
axs = [ax0, ax1]
label = ['(A)', '(B)']
y_pos = [1.25, 1.1]
for n, ax in enumerate(axs):
    ax.text(-0.1, y_pos[n], label[n], transform=ax.transAxes,
            size=11, weight='bold')


bars = ax1.bar(np.arange(8), simulated_rates,
               align='center', color=sim_colors[1])

for i in [0, 2, 4, 6]:
    bars[i].set_color(sim_colors[0])

nnmt_handle = ax1.scatter(np.arange(8), firing_rates, marker='X',
                          color=thy_color, s=50, zorder=10)
ax1.set_xticks(np.arange(8))
ax1.set_xticklabels(['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I'])
ax1.set_yticks([1, 3, 5, 7])
ax1.set_ylabel(r'$\nu\,(1/s)$')

plt.legend([bars[0], nnmt_handle, bars[1]],
           [None, 'theory', 'simulation'],
           loc='upper left', fontsize=9, ncol=2,
           columnspacing=-2.8, handletextpad=0.2)

# insert sketch using svgutil, try saving as pdf using inkscape
if insert_sketch:
    sketch_fn = 'figures/microcircuit_sketch.svg'
    plot_fn = 'figures/microcircuit_rates'
    svg_mpl = sg.from_mpl(fig, savefig_kw=dict(transparent=True))
    w_svg, h_svg = svg_mpl.get_size()
    svg_mpl.set_size((w_svg+'pt', h_svg+'pt'))
    svg_sketch = sg.fromfile(sketch_fn).getroot()
    svg_sketch.moveto(x=30, y=10, scale_x=0.56, scale_y=0.56)
    svg_mpl.append(svg_sketch)
    svg_mpl.save(f'{plot_fn}.svg')
    os_return = os.system(f'inkscape --export-pdf={plot_fn}.pdf {plot_fn}.svg')
    if os_return == 0:
        os.remove(f'{plot_fn}.svg')
    else:
        print('Conversion to pdf using inkscape failed, keeping svg...')

ax0.annotate('(sketch)', xy=(0.35, 0.6))
plt.show()
