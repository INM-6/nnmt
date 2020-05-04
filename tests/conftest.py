import pytest
import numpy as np

from .utils import get_required_keys, get_required_params

from lif_meanfield_tools.input_output import load_params
from lif_meanfield_tools import ureg


@pytest.fixture
def all_std_params():
    return dict(mu=1*ureg.mV,
                sigma=1*ureg.mV,
                nu=1*ureg.Hz,
                K=1,
                J=1*ureg.mV,
                j=1*ureg.mV,
                V_th_rel=15*ureg.mV,
                V_0_rel=0*ureg.mV,
                tau_m=20*ureg.s,
                tau_s=1*ureg.s,
                tau_r=1*ureg.s,
                nu_ext=1*ureg.Hz,
                K_ext=1,
                g=1,
                nu_e_ext=1*ureg.Hz,
                nu_i_ext=1*ureg.Hz,
                dimension=1,
                omega=1*ureg.Hz,
                omegas=[1*ureg.Hz])


@pytest.fixture
def std_params(request, all_std_params):
    """Returns set of standard params needed for all tests."""
    return get_required_params(request.cls.func, all_std_params)


all_pos_keys = ['nu',
                'K',
                'tau_m',
                'tau_s',
                'tau_r',
                'nu_ext',
                'K_ext',
                'nu_e_ext',
                'nu_i_ext',
                'dimension',
                'sigma']


fixture_path = 'tests/fixtures/'

# noise driven regime mu < V_th
# parameters taken from microcircuit example
# need to rename, because results are saved under different key than args need
_params_noise_driven_regime = load_params(
    '{}noise_driven_regime.yaml'.format(fixture_path))
_params_noise_driven_regime['mu'] = _params_noise_driven_regime['mean_input']
_params_noise_driven_regime['sigma'] = _params_noise_driven_regime['std_input']
_params_noise_driven_regime['nu'] = _params_noise_driven_regime['firing_rates']

# regime in which negative firing rates occured once
# parameters taken from circuit in which lmt returned negative rates
_params_negative_firing_rate_regime = load_params(
    '{}negative_firing_rate_regime.yaml'.format(fixture_path))
_params_negative_firing_rate_regime['mu'] = (
    _params_negative_firing_rate_regime['mean_input'])
_params_negative_firing_rate_regime['sigma'] = (
    _params_negative_firing_rate_regime['std_input'])
_params_negative_firing_rate_regime['nu'] = (
    _params_negative_firing_rate_regime['firing_rates'])

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

ids_all_regimes = ['noise_driven_regime',
                    'negative_firing_rate_regime']
                    # 'mean_driven_regime']

_params_all_regimes = [_params_noise_driven_regime,
                       _params_negative_firing_rate_regime, ]
                       # params_mean_driven_regime]

tf_shift_noise_path = ('{}transfer_function_shift_noise_driven.npy'
                       ).format(fixture_path)
tf_shift_neg_rate_path = ('{}transfer_function_shift_negative_firing_rate.npy'
                          ).format(fixture_path)
tf_taylor_noise_path = ('{}transfer_function_taylor_noise_driven.npy'
                        ).format(fixture_path)
tf_taylor_neg_rate_path = ('{}transfer_function_taylor_negative_firing_rate'
                           '.npy').format(fixture_path)

_transfer_function_shift = [np.load(tf_shift_noise_path),
                            np.load(tf_shift_neg_rate_path)]
_transfer_function_taylor = [np.load(tf_taylor_noise_path),
                             np.load(tf_taylor_neg_rate_path)]

all_params = [dict(params, **dict(tf_shift=tf_s, tf_taylor=tf_t))
              for params, tf_s, tf_t in zip(_params_all_regimes,
                                            _transfer_function_shift,
                                            _transfer_function_taylor)
              ]


def pytest_generate_tests(metafunc):
    """Define parametrization schemes for pos_keys and output_test_fixtures."""
    func = metafunc.cls.func

    if "pos_keys" in metafunc.fixturenames:
        # test every pos_key required by func as arg separately
        pos_keys = get_required_keys(func, all_pos_keys)
        metafunc.parametrize("pos_keys", pos_keys)

    elif "output_test_fixtures" in metafunc.fixturenames:
        # test every parameter regime seperately using all_params
        params = [get_required_params(func, params) for params in all_params]

        output_key = metafunc.cls.output_key
        output = [params[output_key] for params in all_params]

        fixtures = [dict(output=output, params=params) for output, params
                    in zip(output, params)]

        metafunc.parametrize("output_test_fixtures", fixtures,
                             ids=ids_all_regimes)
