import pytest
import numpy as np
from inspect import signature

import lif_meanfield_tools as lmt
from lif_meanfield_tools.input_output import load_h5
from lif_meanfield_tools import ureg


config_path = 'tests/fixtures/config/'
fixture_path = 'tests/fixtures/data/'


# list of all always positive arguments
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
                'sigma',
                'tau_m',
                'tau_s',
                'tau_r',
                'nu_ext',
                ]


regimes = ['noise_driven', 'negative_firing_rate']
key_pairs = (('mean_input', 'mu'),
             ('std_input', 'sigma'),
             ('firing_rates', 'nu'),
             ('delay_dist', 'delay_dist_matrix'))
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


def get_required_keys(func, all_keys):
    """Checks arguments of func and returns corresponding parameters."""
    arg_keys = list(signature(func).parameters)
    required_keys = [key for key in all_keys if key in arg_keys]
    return required_keys


def get_required_params(func, all_params):
    """Checks arguments of func and returns corresponding parameters."""
    required_keys = list(signature(func).parameters)
    required_params = {k: v for k, v in all_params.items()
                       if k in required_keys}
    return required_params


@pytest.fixture
def network():
    """Returns standard microcircuit network."""
    network = lmt.Network(
        network_params=(config_path + 'network_params_microcircuit.yaml'),
        analysis_params=(config_path + 'analysis_params_test.yaml')
        )
    return network


@pytest.fixture
def network_params_yaml():
    """Returns standard microcircuit network params."""
    return config_path + 'network_params_microcircuit.yaml'


@pytest.fixture
def analysis_params_yaml():
    """Returns some standard analysis params."""
    return config_path + 'analysis_params_test.yaml'


def calc_dep_params(params):
    """
    Calcs dependend parameters derived from params and updates params.

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
    W = np.ones((dim, dim)) * params['w']
    W[1:dim:2] *= -params['g']
    W = np.transpose(W)
    params['W'] = W

    # weight matrix in mV (voltage)
    params['J'] = (tau_s_div_C * params['W']).to(ureg.mV)

    # delay matrix
    D = np.ones((dim, dim)) * params['d_e']
    D[1:dim:2] = np.ones(dim) * params['d_i']
    D = np.transpose(D)
    params['Delay'] = D

    # delay standard deviation matrix
    D = np.ones((dim, dim)) * params['d_e_sd']
    D[1:dim:2] = np.ones(dim) * params['d_i_sd']
    D = np.transpose(D)
    params['Delay_sd'] = D


@pytest.fixture
def all_std_params():
    # standard params from first two populations of microcircuit
    firing_rates = np.array([0.71, 2.75]) * ureg.Hz
    mean_input = np.array([3.30, 7.03]) * ureg.mV
    std_input = np.array([6.19, 5.11]) * ureg.mV
    params = dict(C=250 * ureg.pF,
                  K=np.array([[2199, 1079], [2990, 860]]),
                  K_ext=np.array([1600, 1500]),
                  N=np.array([20683, 5834]),
                  V_th_abs=-50 * ureg.mV,
                  V_0_abs=-65 * ureg.mV,
                  d_e=1.5 * ureg.ms,
                  d_i=0.75 * ureg.ms,
                  d_e_sd=0.75 * ureg.ms,
                  d_i_sd=0.375 * ureg.ms,
                  delay_dist=None,
                  delay_dist_matrix=([[complex(1.000, -0.005),
                                       complex(1.000, -0.002)],
                                      [complex(1.000, -0.005),
                                       complex(1.000, -0.002)]]
                                     ) * ureg.dimensionless,
                  firing_rates=firing_rates,
                  g=4,
                  label='microcircuit',
                  matrix='MH',
                  mean_input=mean_input,
                  mu=mean_input,
                  mu_set=np.array([3, 7]) * ureg.mV,
                  nu=firing_rates,
                  nu_ext=8 * ureg.Hz,
                  nu_e_ext=np.array([0, 0]) * ureg.Hz,
                  nu_i_ext=np.array([0, 0]) * ureg.Hz,
                  populations=['23E', '23I'],
                  sigma=std_input,
                  sigma_set=np.array([6, 5]) * ureg.mV,
                  std_input=std_input,
                  tau_m=10 * ureg.s,
                  tau_s=0.5 * ureg.s,
                  tau_r=2.0 * ureg.s,
                  transfer_function=[complex(0.653, -0.101),
                                     complex(1.811, -0.228),
                                     ] * ureg.Hz / ureg.mV,
                  omega=20 * ureg.Hz,
                  omegas=np.array([20]) * ureg.Hz,
                  w=87.8 * ureg.pA
                  )
    calc_dep_params(params)
    return params


@pytest.fixture
def std_params(request, all_std_params):
    """
    Returns set of standard params needed by requesting tested function.
    
    For using this fixture, the function test class needs to have a class
    attribute `func`, which is the tested function as a staticmethod.
    """
    return get_required_params(request.cls.func, all_std_params)


@pytest.fixture
def std_params_tf(std_params):
    """Returns set of standard params needed for transfer function tests."""
    std_params['dimension'] = 1
    std_params['mu'] = np.array([1, 2]) * ureg.mV
    std_params['sigma'] = np.array([1, 2]) * ureg.mV
    return std_params


@pytest.fixture
def std_params_eval_spectra(request, all_std_params):
    """Returns set of standard params needed eigenvalue tests."""
    all_std_params['quantity'] = 'eigvals'
    return get_required_params(request.cls.func, all_std_params)


def get_output_for_keys_of_metafunc(metafunc, results, params):
    """
    Returns output fixtures for output keys of the requesting test class.
    
    If a test class has `output_key` or `output_keys` as class attribute,
    this function returns a list of the outputs for the given keys.
    """
    if hasattr(metafunc.cls, 'output_key'):
        output_key = metafunc.cls.output_key
        return [result[output_key] for result in results]
    elif hasattr(metafunc.cls, 'output_keys'):
        output_keys = metafunc.cls.output_keys
        return [[result[output_key] for output_key in output_keys]
                for result in results]
    else:
        return [[] for i in range(len(params))]


def pytest_generate_tests(metafunc, all_params=all_params, results=results,
                          ids_all_regimes=ids_all_regimes):
    """
    Special pytest function defining parametrizations for certain fixtures.
    
    `pos_keys`:
    If a test requires all positive keys contained in the list of arguments of
    the tested function, the corresponding function test class needs to have a
    class attribute `func`, which is the tested function as a staticmethod.
    The pos keys are tested one after each other as a parametrization.
    
    `output_test_fixtures`:
    If a test requires input arguments and outputs in different regimes for
    comparison with the return values of the tested function, the corresponding
    function test class needs a class attribute `func`, the tested function as
    a staticmethod, and either a `output_key` attribute or a `output_keys`
    attribute. The different parameter regimes are then tested one after each
    other as a parametrization. There are some functions that need a special
    treatment, like the sensitivity_measure (see code).
    """
    # check if requesting test class has class attribute func
    if hasattr(metafunc.cls, 'func'):
        func = metafunc.cls.func
    # if it does not, just return and don't parametrize
    else:
        return None
    
    if "pos_keys" in metafunc.fixturenames:
        # test every pos_key required by func as arg separately
        pos_keys = get_required_keys(func, all_pos_keys)
        metafunc.parametrize("pos_keys", pos_keys)

    elif "output_test_fixtures" in metafunc.fixturenames:

        if 'prop_inv' in metafunc.function.__name__:
            # take out negative_firing_rate regime because prop is singular
            singular_regime = 'negative_firing_rate'
            indices = [i for i, params in enumerate(all_params)
                       if params['regime'] != singular_regime]
            all_params = [all_params[i] for i in indices]
            results = [results[i] for i in indices]
            ids_all_regimes = [ids_all_regimes[i] for i in indices]

        # test every parameter regime seperately using all_params
        params = [get_required_params(func, dict(params, **results))
                  for params, results in zip(all_params, results)]

        output = get_output_for_keys_of_metafunc(metafunc, results, params)

        if 'sensitivity_measure' in metafunc.cls.__name__:
            for param, result in zip(params, results):
                # sensitivity measure requires special transfer function as arg
                param['transfer_function'] = result['transfer_function_'
                                                    'single'][0]
                # sensitivity measure requires special delay_dist_matrix as arg
                param['delay_dist_matrix'] = result['delay_dist_single'][0]

        fixtures = [dict(output=output, params=params) for output, params
                    in zip(output, params)]

        metafunc.parametrize("output_test_fixtures", fixtures,
                             ids=ids_all_regimes)
