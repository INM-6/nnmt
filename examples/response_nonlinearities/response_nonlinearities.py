"""
Response nonlinearities
=======================

In this example, we reproduce the different types of response nonlinearities of
an EI network that were uncovered in :cite:t:`sanzeni2020`. To this end, we
need to determine the self-consistent rates of EI networks with specific
indegrees and synaptic weights for changing external input.

Most of this script handles all the necessary parameters, the relevant
calculation is performed by the function :meth:`nnmt.lif.delta._firing_rates`.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nnmt.lif.delta import _firing_rates

# try loading svgutils to add the network sketch
try:
    import os
    import svgutils.transform as sg
    insert_sketch = True
except ImportError:
    insert_sketch = False

# use matplotlib style file
plt.style.use('frontiers.mplstyle')


###############################################################################
# First, we define the common parameters for all networks: the time constants
# and the reset and threshold voltage.
params_all = dict(
    # time constants in s
    tau_m=20.*1e-3, tau_r=2.*1e-3,
    # reset and threshold voltage relative to leak in mV
    V_0_rel=10., V_th_rel=20.,
)


###############################################################################
# Next, we define the indegrees and the synaptic weights corresponding to the
# different nonlinearities.

# Saturation-driven nonlinearity
g_E, g_I, a_E, a_I = 8, 7, 4, 2
params_sdn = dict(
    J=np.array([[0.2, -g_E*0.2], [0.2, -g_I*0.2]]),
    K=np.array([[400., 100.], [400., 100.]]),
    J_ext=np.array([0.2, 0.2]),
    K_ext=np.array([a_E*400., a_I*400.])
)
# Saturation-driven multisolution
g_E, g_I, a_E, a_I = 2.08, 1.67, 1, 1
params_sdm = dict(
    J=np.array([[0.2, -g_E*0.2], [2.4*0.2/2.5, -g_I*2.4*0.2/2.5]]),
    K=np.array([[400., 100.], [400., 100.]]),
    J_ext=np.array([0.2, 2.4*0.2/2.5]),
    K_ext=np.array([a_E*400., a_I*400.])
)
# Response-onset supersaturation
g_E, g_I, a_E, a_I = 4.5, 2.9, 1, 1
params_ros = dict(
    J=np.array([[0.2, -g_E*0.2], [0.2, -g_I*0.2]]),
    K=np.array([[400., 100.], [400., 100.]]),
    J_ext=np.array([0.2, 0.2]),
    K_ext=np.array([a_E*400., a_I*400.])
)
# Mean-driven multisolution
g_E, g_I, a_E, a_I = 4.1, 2.46, 1, 0.2
params_mdm = dict(
    K=np.array([[800., 200.], [400., 100.]]),
    J=np.array([[0.2, -g_E*0.2], [0.2, -g_I*0.2]]),
    J_ext=np.array([0.2, 0.2]),
    K_ext=np.array([a_E*800., a_I*400.])
)
# Noise-driven multisolution
g_E, g_I, a_E, a_I = 7, 6, 1, 0.7
params_ndm = dict(
    K=np.array([[400., 100.], [400., 100.]]),
    J=np.array([[0.5, -g_E*0.5], [0.5, -g_I*0.5]]),
    J_ext=np.array([0.5, 0.5]),
    K_ext=np.array([a_E*400., a_I*400.])
)


###############################################################################
# We introduce a helper function to handle the parameters. The firing rates
# are determined using the :meth:`nnmt.lif.delta._firing_rates` function.
def solve_theory(params, nu_0, nu_ext_min, nu_ext_max, nu_ext_steps, method):
    # combine common and specific parameters
    params.update(params_all)
    # create an array with all external rates and an array for the results
    nu_ext_arr = np.linspace(nu_ext_min, nu_ext_max, nu_ext_steps)
    nu_arr = np.zeros((nu_ext_steps, 2))
    # iterate through the ext. rates and determine the self-consistent rates
    for i, nu_ext in enumerate(nu_ext_arr):
        try:
            nu_arr[i] = _firing_rates(nu_0=nu_0, nu_ext=nu_ext,
                                      fixpoint_method=method, **params)
        except RuntimeError:
            # set non-convergent solutions to nan for convenience
            nu_arr[i] = (np.nan, np.nan)
    return nu_ext_arr, nu_arr


###############################################################################
# Now, we calculate the firing rate for each nonlinearity. By default, we use
# the `ODE` method. If this does not converge, we use `LSTSQ`.
print('Calculating self-consitent rates...')
print('Saturation-driven nonlinearity...')
nu_ext_sdn, nu_sdn = solve_theory(params_sdn, (0, 0), 1, 100, 50,
                                  method='ODE')
print('Saturation-driven multisolution...')
nu_ext_sdm_a, nu_sdm_a = solve_theory(params_sdm, (0, 0), 1, 9, 10,
                                      method='ODE')
nu_ext_sdm_b, nu_sdm_b = solve_theory(params_sdm, (500, 500), 1, 100, 50,
                                      method='ODE')
nu_ext_sdm_c, nu_sdm_c = solve_theory(params_sdm, (10, 10), 1, 20, 10,
                                      method='LSTSQ')
nu_ext_sdm_d, nu_sdm_d = solve_theory(params_sdm, (100, 100), 1, 20, 10,
                                      method='LSTSQ')
print('Response-onset supersaturation...')
nu_ext_ros_a, nu_ros_a = solve_theory(params_ros, (0, 0), 0.5, 50, 50,
                                      method='ODE')
nu_ext_ros_b, nu_ros_b = solve_theory(params_ros, (10, 10), 7.5, 12.5, 50,
                                      method='ODE')
print('Mean-driven multisolution...')
nu_ext_mdm_a, nu_mdm_a = solve_theory(params_mdm, (0, 0), 0.1, 5, 25,
                                      method='LSTSQ')
nu_ext_mdm_b, nu_mdm_b = solve_theory(params_mdm, (50, 50), 0.1, 10, 50,
                                      method='LSTSQ')
nu_ext_mdm_c, nu_mdm_c = solve_theory(params_mdm, (10, 0), 0.1, 5, 25,
                                      method='LSTSQ')
print('Noise-driven multisolution...')
nu_ext_ndm_a, nu_ndm_a = solve_theory(params_ndm, (0, 0), 0.05, 5, 50,
                                      method='ODE')
nu_ext_ndm_b, nu_ndm_b = solve_theory(params_ndm, (10, 10), 0.05, 5, 50,
                                      method='ODE')
nu_ext_ndm_c, nu_ndm_c = solve_theory(params_ndm, (5, 4), 0.05, 5, 50,
                                      method='LSTSQ')
nu_ext_ndm_d, nu_ndm_d = solve_theory(params_ndm, (2, 0), 0.05, 5, 50,
                                      method='LSTSQ')


###############################################################################
# Finally, we plot the result. Again, we introduce a helper function to handle
# the parameters. Panel (A) contains a sketch of the network that can only be
# added in the pdf output but is not shown in `plt.show()`.
def plot_rates(ax, nu_ext_arr_lst, nu_arr_lst, xmax, ymax, xlabel, ylabel,
               title, colors, label, label_prms):
    ax.set_prop_cycle(color=colors)
    for i, nu_arr in enumerate(nu_arr_lst):
        ax.plot(nu_ext_arr_lst[i], nu_arr, 'o')
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xticks((0, xmax/2, xmax))
    ax.set_yticks((0, ymax/2, ymax))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(s=label, transform=ax.transAxes, **label_prms)


print('Plotting...')
fig = plt.figure(figsize=(7.08661, 2.95276),  # two column figure, 180mm wide
                 constrained_layout=True)
gs = gridspec.GridSpec(2, 3, figure=fig)
label_prms = dict(x=-0.3, y=1.45, fontsize=10, fontweight='bold',
                  va='top', ha='right')
colors = ['#4c72b0', '#c44e52']
# Sketch
ax_sketch = fig.add_subplot(gs[0, 0])
ax_sketch.axis('off')
ax_sketch.set_title('network\nsketch')
ax_sketch.text(s='A', transform=ax_sketch.transAxes, **label_prms)
# Saturation-driven nonlinearity
plot_rates(fig.add_subplot(gs[0, 1]), [nu_ext_sdn], [nu_sdn], 100, 500,
           '', r'rate $\nu$ (1/s)', 'saturation-driven\nnonlinearity',
           colors, 'B', label_prms)
# Saturation-driven multisolution
plot_rates(fig.add_subplot(gs[0, 2]),
           [nu_ext_sdm_a, nu_ext_sdm_b, nu_ext_sdm_c, nu_ext_sdm_d],
           [nu_sdm_a, nu_sdm_b, nu_sdm_c, nu_sdm_d], 100, 500,
           '', '', 'saturation-driven\nmulti-solution',
           colors, 'C', label_prms)
# Response-onset supersaturation
plot_rates(fig.add_subplot(gs[1, 0]), [nu_ext_ros_a, nu_ext_ros_b],
           [nu_ros_a, nu_ros_b], 50, 5,
           r'external rate $\nu_X$ (1/s)', r'rate $\nu$ (1/s)',
           'response-onset\nsupersaturation', colors, 'D', label_prms)
# Mean-driven multisolution
plot_rates(fig.add_subplot(gs[1, 1]),
           [nu_ext_mdm_a, nu_ext_mdm_b, nu_ext_mdm_c],
           [nu_mdm_a, nu_mdm_b, nu_mdm_c], 10, 50,
           r'external rate $\nu_X$ (1/s)', '', 'mean-driven\nmulti-solution',
           colors, 'E', label_prms)
# Noise-driven multisolution
plot_rates(fig.add_subplot(gs[1, 2]),
           [nu_ext_ndm_a, nu_ext_ndm_b, nu_ext_ndm_c, nu_ext_ndm_d],
           [nu_ndm_a, nu_ndm_b, nu_ndm_c, nu_ndm_d], 5, 10,
           r'external rate $\nu_X$ (1/s)', '', 'noise-driven\nmulti-solution',
           colors, 'F', label_prms)

# insert sketch using svgutil, try saving as pdf using inkscape
if insert_sketch:
    sketch_fn = 'brunel_sketch.svg'
    plot_fn = 'response_nonlinearities'
    svg_mpl = sg.from_mpl(fig, savefig_kw=dict(transparent=True))
    w_svg, h_svg = svg_mpl.get_size()
    svg_mpl.set_size((w_svg+'pt', h_svg+'pt'))
    svg_sketch = sg.fromfile(sketch_fn).getroot()
    svg_sketch.moveto(x=25, y=30, scale_x=1.0)
    svg_mpl.append(svg_sketch)
    svg_mpl.save(f'{plot_fn}.svg')
    os_return = os.system(f'inkscape --export-pdf={plot_fn}.pdf {plot_fn}.svg')
    if os_return == 0:
        os.remove(f'{plot_fn}.svg')
    else:
        print('Conversion to pdf using inkscape failed, keeping svg...')

# show figure (without sketch)
ax_sketch.annotate('(sketch)', xy=(0., 0.5))
plt.show()
