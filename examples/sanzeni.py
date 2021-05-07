import numpy as np
import matplotlib.gridspec as gridspec

from lif_meanfield_tools import ureg
from lif_meanfield_tools.meanfield_calcs import firing_rates

import matplotlib.pyplot as plt
plt.style.use('frontiers.mplstyle')


# ===== parameters =====
nu_ext_steps = 50  # number of external rates (x-axis)
# common parameters for all plots
params_all = dict(
    dimension=2,  # E and I
    # time constants
    tau_m=20*ureg.ms, tau_s=0*ureg.ms, tau_r=2*ureg.ms,
    # reset and threshold voltage relative to leak
    V_0_rel=10*ureg.mV, V_th_rel=20*ureg.mV,
    # no additional external input
    g=1, nu_e_ext=0*ureg.Hz, nu_i_ext=0*ureg.Hz,
    # method irrelevant because tau_s=0
    method='shift'
)
# Saturation-driven nonlinearity
g_E, g_I, a_E, a_I = 8, 7, 4, 2
params_sdn = dict(
    J=np.array([[0.2, -g_E*0.2], [0.2, -g_I*0.2]])*ureg.mV,
    K=np.array([[400., 100.], [400., 100.]]),
    j=np.array([0.2, 0.2])*ureg.mV,
    K_ext=np.array([a_E*400., a_I*400.])
)
# Saturation-driven multisolution
g_E, g_I, a_E, a_I = 2.08, 1.67, 1, 1
params_sdm = dict(
    J=np.array([[0.2, -g_E*0.2], [2.4*0.2/2.5, -g_I*2.4*0.2/2.5]])*ureg.mV,
    K=np.array([[400., 100.], [400., 100.]]),
    j=np.array([0.2, 2.4*0.2/2.5])*ureg.mV,
    K_ext=np.array([a_E*400., a_I*400.])
)
# Response-onset supersaturation
g_E, g_I, a_E, a_I = 4.5, 2.9, 1, 1
params_ros = dict(
    J=np.array([[0.2, -g_E*0.2], [0.2, -g_I*0.2]])*ureg.mV,
    K=np.array([[400., 100.], [400., 100.]]),
    j=np.array([0.2, 0.2])*ureg.mV,
    K_ext=np.array([a_E*400., a_I*400.])
)
# Mean-driven multisolution
g_E, g_I, a_E, a_I = 4.1, 2.46, 1, 0.2
params_mdm = dict(
    K=np.array([[800., 200.], [400., 100.]]),
    J=np.array([[0.2, -g_E*0.2], [0.2, -g_I*0.2]])*ureg.mV,
    j=np.array([0.2, 0.2])*ureg.mV,
    K_ext=np.array([a_E*800., a_I*400.])
)
# Noise-driven multisolution
g_E, g_I, a_E, a_I = 7, 6, 1, 0.7
params_ndm = dict(
    K=np.array([[400., 100.], [400., 100.]]),
    J=np.array([[0.5, -g_E*0.5], [0.5, -g_I*0.5]])*ureg.mV,
    j=np.array([0.5, 0.5])*ureg.mV, K_ext=np.array([a_E*400., a_I*400.])
)


# ===== self-consistent rates =====
def solve_theory(params, nu_0, nu_ext_min, nu_ext_max, nu_ext_steps,
                 fp_iteration=True, print_rates=False):
    params.update(params_all)
    nu_ext_arr = np.linspace(nu_ext_min, nu_ext_max, nu_ext_steps)*ureg.Hz
    nu_arr = np.zeros((nu_ext_steps+1, 2))*ureg.Hz
    for i, nu_ext in enumerate(nu_ext_arr):
        nu_arr[i+1] = firing_rates(nu_0=nu_0*ureg.Hz, nu_ext=nu_ext,
                                   fp_iteration=fp_iteration, **params)
        if print_rates:
            print(nu_ext, nu_arr[i+1])
    nu_ext_arr = np.insert(nu_ext_arr, 0, 0)
    return nu_ext_arr, nu_arr


print('Calculating self-consitent rates...')
# TODO: optimize intervals & find more unstable branches
print('Saturation-driven nonlinearity...')
nu_ext_sdn, nu_sdn = solve_theory(params_sdn, (0, 0), 1, 100,
                                  nu_ext_steps)
print('Saturation-driven multisolution...')
nu_ext_sdm_a, nu_sdm_a = solve_theory(params_sdm, (0, 0), 1, 9,
                                      nu_ext_steps)
nu_ext_sdm_b, nu_sdm_b = solve_theory(params_sdm, (500, 500), 1, 100,
                                      nu_ext_steps)
nu_ext_sdm_c, nu_sdm_c = solve_theory(params_sdm, (10, 10), 1, 20,
                                      nu_ext_steps, fp_iteration=False)
nu_ext_sdm_d, nu_sdm_d = solve_theory(params_sdm, (100, 100), 1, 20,
                                      nu_ext_steps, fp_iteration=False)
print('Response-onset supersaturation...')
nu_ext_ros_a, nu_ros_a = solve_theory(params_ros, (0, 0), 0.5, 50,
                                      nu_ext_steps)
nu_ext_ros_b, nu_ros_b = solve_theory(params_ros, (10, 10), 7.5, 12.5,
                                      nu_ext_steps)
print('Mean-driven multisolution...')
nu_ext_mdm_a, nu_mdm_a = solve_theory(params_mdm, (0, 0), 0.1, 5,
                                      nu_ext_steps, fp_iteration=False)
nu_ext_mdm_b, nu_mdm_b = solve_theory(params_mdm, (50, 50), 0.1, 10,
                                      nu_ext_steps, fp_iteration=False)
nu_ext_mdm_c, nu_mdm_c = solve_theory(params_mdm, (10, 0), 0.1, 5,
                                      nu_ext_steps, fp_iteration=False)
print('Noise-driven multisolution...')
nu_ext_ndm_a, nu_ndm_a = solve_theory(params_ndm, (0, 0), 0.05, 5,
                                      nu_ext_steps)
nu_ext_ndm_b, nu_ndm_b = solve_theory(params_ndm, (10, 10), 0.05, 5,
                                      nu_ext_steps)
nu_ext_ndm_c, nu_ndm_c = solve_theory(params_ndm, (5, 4), 0.05, 5,
                                      nu_ext_steps, fp_iteration=False)
nu_ext_ndm_d, nu_ndm_d = solve_theory(params_ndm, (2, 0), 0.05, 5,
                                      nu_ext_steps, fp_iteration=False)


# ===== figure =====
def plot_rates(ax, nu_ext_arr_lst, nu_arr_lst, xmax, ymax, xlabel, ylabel,
               title, colors, label, label_prms):
    ax.set_prop_cycle(color=colors)
    for i, nu_arr in enumerate(nu_arr_lst):
        ax.plot(nu_ext_arr_lst[i], nu_arr, 'o')
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xticks((0, xmax))
    ax.set_yticks((0, ymax))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(s=label, transform=ax.transAxes, **label_prms)


print('Plotting...')
fig = plt.figure(figsize=(7.08661, 1.45),  # two column figure, 180mm wide
                 constrained_layout=True)
gs = gridspec.GridSpec(1, 5, figure=fig)
label_prms = dict(x=-0.3, y=1.3, fontsize=10, fontweight='bold',
                  va='top', ha='right')
colors = ['#c44e52', '#4c72b0']
# Saturation-driven nonlinearity
plot_rates(fig.add_subplot(gs[0, 0]), [nu_ext_sdn], [nu_sdn], 100, 500,
           r'$\nu_X$ [spks/s]', r'$\nu$ [spks/s]', 'SDN',
           colors, '(A)', label_prms)
# Saturation-driven multisolution
plot_rates(fig.add_subplot(gs[0, 1]),
           [nu_ext_sdm_a, nu_ext_sdm_b, nu_ext_sdm_c, nu_ext_sdm_d],
           [nu_sdm_a, nu_sdm_b, nu_sdm_c, nu_sdm_d], 100, 500,
           r'$\nu_X$ [spks/s]', '', 'SDM',
           colors, '(B)', label_prms)
# Response-onset supersaturation
plot_rates(fig.add_subplot(gs[0, 2]), [nu_ext_ros_a, nu_ext_ros_b],
           [nu_ros_a, nu_ros_b], 50, 5,
           r'$\nu_X$ [spks/s]', '', 'ROS',
           colors, '(C)', label_prms)
# Mean-driven multisolution
plot_rates(fig.add_subplot(gs[0, 3]),
           [nu_ext_mdm_a, nu_ext_mdm_b, nu_ext_mdm_c],
           [nu_mdm_a, nu_mdm_b, nu_mdm_c], 10, 50,
           r'$\nu_X$ [spks/s]', '', 'MDM',
           colors, '(D)', label_prms)
# Noise-driven multisolution
plot_rates(fig.add_subplot(gs[0, 4]),
           [nu_ext_ndm_a, nu_ext_ndm_b, nu_ext_ndm_c, nu_ext_ndm_d],
           [nu_ndm_a, nu_ndm_b, nu_ndm_c, nu_ndm_d], 5, 10,
           r'$\nu_X$ [spks/s]', '', 'NDM',
           colors, '(E)', label_prms)
# save and show
plt.savefig('sanzeni_fig8.png', dpi=300)
plt.show()
