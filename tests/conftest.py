import pytest
import numpy as np

from .utils import get_required_keys, get_required_params

from lif_meanfield_tools.input_output import load_params, load_h5
from lif_meanfield_tools import ureg


def calc_dep_params(params):
    """
    Calcs dependend parameters derived from params.

    Taken from network.py.
    """

    # calculate dimension of system
    dim = len(params['populations'])
    params['dimension'] = dim

    # reset reference potential to 0
    params['V_0_rel'] = 0 * ureg.mV
    params['V_th_rel'] = (params['V_th_abs'] - params['V_0_abs'])

    # convert weights in pA (current) to weights in mV (voltage)
    tau_s_div_C = params['tau_s'] / params['C']
    params['j'] = (tau_s_div_C * params['w']).to(ureg.mV)

    # weight matrix in pA (current)
    W = np.ones((dim, dim))*params['w']
    W[1:dim:2] *= -params['g']
    W = np.transpose(W)
    params['W'] = W

    # weight matrix in mV (voltage)
    params['J'] = (tau_s_div_C * params['W']).to(ureg.mV)

    # delay matrix
    D = np.ones((dim, dim))*params['d_e']
    D[1:dim:2] = np.ones(dim)*params['d_i']
    D = np.transpose(D)
    params['Delay'] = D

    # delay standard deviation matrix
    D = np.ones((dim, dim))*params['d_e_sd']
    D[1:dim:2] = np.ones(dim)*params['d_i_sd']
    D = np.transpose(D)
    params['Delay_sd'] = D


@pytest.fixture
def all_std_params():

    # standard params from first two populations of microcircuit
    params = dict(C=250*ureg.pF,
                  K=np.array([[2199, 1079], [2990, 860]]),
                  K_ext=np.array([1600, 1500]),
                  N=np.array([20683, 5834]),
                  V_th_abs=-50*ureg.mV,
                  V_0_abs=-65*ureg.mV,
                  d_e=1.5*ureg.ms,
                  d_i=0.75*ureg.ms,
                  d_e_sd=0.75*ureg.ms,
                  d_i_sd=0.375*ureg.ms,
                  delay_dist=None,
                  g=4,
                  label='microcircuit',
                  mu=np.array([3.30, 7.03])*ureg.mV,
                  nu=np.array([0.71, 2.75])*ureg.Hz,
                  nu_ext=8*ureg.Hz,
                  nu_e_ext=np.array([0, 0])*ureg.Hz,
                  nu_i_ext=np.array([0, 0])*ureg.Hz,
                  populations=['23E', '23I'],
                  sigma=np.array([6.19, 5.11])*ureg.mV,
                  tau_m=10*ureg.s,
                  tau_s=0.5*ureg.s,
                  tau_r=2.0*ureg.s,
                  omega=20*ureg.Hz,
                  omegas=np.array([20])*ureg.Hz,
                  w=87.8*ureg.pA
                  )

    calc_dep_params(params)

    return params


@pytest.fixture
def std_params(request, all_std_params):
    """Returns set of standard params needed for all tests."""
    return get_required_params(request.cls.func, all_std_params)


@pytest.fixture
def std_params_tf(std_params):
    """Returns set of standard params needed for all tests."""
    std_params['dimension'] = 1
    std_params['mu'] = np.array([1, 2])*ureg.mV
    std_params['sigma'] = np.array([1, 2])*ureg.mV
    return std_params


all_pos_keys = ['C',
                'K',
                'K_ext',
                'N',
                'd_e',
                'd_i',
                'd_e_sd',
                'd_i_sd',
                'dimension',
                'g',
                'nu',
                'nu_ext',
                'nu_e_ext',
                'nu_i_ext',
                'sigma'
                'tau_m',
                'tau_s',
                'tau_r',
                'nu_ext',
                ]


fixture_path = 'tests/fixtures/'
regimes = ['noise_driven', 'negative_firing_rate']
key_pairs = (('mean_input', 'mu'),
             ('std_input', 'sigma'),
             ('firing_rates', 'nu'))
all_params = []
results = []
ids_all_regimes = []
for i, regime in enumerate(regimes):
    fixs = load_h5('{}{}_regime.h5'.format(fixture_path, regime))
    all_params.append(dict(fixs['network_params'], **fixs['analysis_params']))
    result = fixs['results']
    # rename some keys, because some functions require diferent arg names
    for old_key, new_key in key_pairs:
        result[new_key] = result[old_key]
    results.append(result)
    ids_all_regimes.append('{}_regime'.format(regime))

# noise driven regime mu < V_th
# parameters taken from microcircuit example
# need to rename, because results are saved under different key than args need
# _params_noise_driven_regime = load_h5(
#     '{}noise_driven_regime.h5'.format(fixture_path))
# _params_noise_driven_regime['mu'] = _params_noise_driven_regime['mean_input']
# _params_noise_driven_regime['sigma'] = _params_noise_driven_regime['std_input']
# _params_noise_driven_regime['nu'] = _params_noise_driven_regime['firing_rates']
#
# # regime in which negative firing rates occured once
# # parameters taken from circuit in which lmt returned negative rates
# _params_negative_firing_rate_regime = load_h5(
#     '{}negative_firing_rate_regime.h5'.format(fixture_path))
# _params_negative_firing_rate_regime['mu'] = (
#     _params_negative_firing_rate_regime['mean_input'])
# _params_negative_firing_rate_regime['sigma'] = (
#     _params_negative_firing_rate_regime['std_input'])
# _params_negative_firing_rate_regime['nu'] = (
#     _params_negative_firing_rate_regime['firing_rates'])

# # mean driven regime mu > V_th
# # parameters taken from adjusted microcircuit example
# params_mean_driven_regime = load_params(
#                                 'tests/unit/fixtures/mean_driven_regime.yaml')
# params_mean_driven_regime['nu'] = params_mean_driven_regime['firing_rates']

# mus =  [741.89455754, 21.24112709, 35.86521795, 40.69297877, 651.19761921] * ureg.mV
# sigmas = [39.3139564, 6.17632725, 9.79196704, 10.64437979, 37.84928217] * ureg.mV
#
# tau_m = 20. * ureg.ms
# tau_s = 0.5 * ureg.ms
# tau_r = 0.5 * ureg.ms
# V_th_rel = 20 * ureg.mV
# V_0_rel = 0 * ureg.mV
#
# tau_ms = np.repeat(tau_m, len(mus))
# tau_ss = np.repeat(tau_s, len(mus))
# tau_rs = np.repeat(tau_r, len(mus))
# V_th_rels = np.repeat(V_th_rel, len(mus))
# V_0_rels = np.repeat(V_0_rel, len(mus))
#
# _parameters_mean_driven_regime =dict(mu=mus, sigma=sigmas,
#                                      tau_m=tau_ms, tau_s=tau_ss,
#                                      tau_r=tau_rs,
#                                      V_th_rel=V_th_rels,
#                                      V_0_rel=V_0_rels)

# ids_all_regimes = ['noise_driven_regime',
#                     'negative_firing_rate_regime']
#                     # 'mean_driven_regime']
#
# _params_all_regimes = [_params_noise_driven_regime,
#                        _params_negative_firing_rate_regime, ]
# #                        # params_mean_driven_regime]
#
# tf_shift_noise_path = ('{}transfer_function_shift_noise_driven.npy'
#                        ).format(fixture_path)
# tf_shift_neg_rate_path = ('{}transfer_function_shift_negative_firing_rate.npy'
#                           ).format(fixture_path)
# tf_taylor_noise_path = ('{}transfer_function_taylor_noise_driven.npy'
#                         ).format(fixture_path)
# tf_taylor_neg_rate_path = ('{}transfer_function_taylor_negative_firing_rate'
#                            '.npy').format(fixture_path)
#
# _transfer_function_shift = [np.load(tf_shift_noise_path),
#                             np.load(tf_shift_neg_rate_path)]
# _transfer_function_taylor = [np.load(tf_taylor_noise_path),
#                              np.load(tf_taylor_neg_rate_path)]

# all_params = [dict(params, **dict(tf_shift=tf_s, tf_taylor=tf_t))
#               for params, tf_s, tf_t in zip(_params_all_regimes,
#                                             _transfer_function_shift,
#                                             _transfer_function_taylor)
#               ]


def pytest_generate_tests(metafunc):
    """Define parametrization schemes for pos_keys and output_test_fixtures."""
    func = metafunc.cls.func

    if "pos_keys" in metafunc.fixturenames:
        # test every pos_key required by func as arg separately
        pos_keys = get_required_keys(func, all_pos_keys)
        metafunc.parametrize("pos_keys", pos_keys)

    elif "output_test_fixtures" in metafunc.fixturenames:
        # test every parameter regime seperately using all_params
        params = [get_required_params(func, dict(params, **results))
                  for params, results in zip(all_params, results)]

        output_key = metafunc.cls.output_key
        output = [result[output_key] for result in results]

        fixtures = [dict(output=output, params=params) for output, params
                    in zip(output, params)]

        metafunc.parametrize("output_test_fixtures", fixtures,
                             ids=ids_all_regimes)
