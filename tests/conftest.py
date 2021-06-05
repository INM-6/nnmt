"""
Special pytest module containing defs of pytest fixtures and parametrizations.

Fixtures:
---------
network: standard microcircuit network with testing analysis params.
network_params: standard microcircuit network params.
analysis_params: some standard test analysis params.
all_std_params: standard params from first two populations of microcircuit.
std_params: set of standard params needed by requesting tested function.
std_params_tf: set of standard params for transfer function tests.
std_params_eval_spectra: set of standard params for eval_spectra tests.

Parametrization Schemes:
------------------------
pos_keys: parametrizes all pos params needed by requesting tested function.
output_test_fixtures: parametrizes needed args and results for tested regimes.
"""

import pytest
import numpy as np
from inspect import signature
import h5py_wrapper as h5

import lif_meanfield_tools as lmt
from lif_meanfield_tools.input_output import load_val_unit_dict_from_h5
from lif_meanfield_tools import ureg
from lif_meanfield_tools.utils import _strip_units


# path to network configuration files and analysis parameters
config_path = 'tests/fixtures/unit/config/'
# path to fixtures
unit_fix_path = 'tests/fixtures/unit/data/'
integration_fix_path = 'tests/fixtures/integration/data/'

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
                'K_ext',
                'tau_m_ext'
                ]

# list of tested parameter regimes for correct output tests
regimes = ['noise_driven', 'mean_driven']

# list of keys that have two different variable names
key_pairs = (('mean_input', 'mu'),
             ('std_input', 'sigma'),
             ('firing_rates', 'nu'))

all_params = []
results = []
ids_all_regimes = []
for i, regime in enumerate(regimes):
    # load fixtures corresponding to regimes defined above
    # standard file name: `<regime_id>_regime.h5`
    fixs = load_val_unit_dict_from_h5('{}{}_regime.h5'.format(unit_fix_path,
                                                              regime))
    result = fixs['results']
    # add regime to results so test functions can use them as output keys
    result['regime'] = regime
    # rename some keys, because some functions require different arg names
    for old_key, new_key in key_pairs:
        result[new_key] = result[old_key]
    # collect params, results and ids in lists
    # combine network and analysis params in all_params
    all_params.append(dict(fixs['network_params'], **fixs['analysis_params']))
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
def unit_fixture_path():
    """Path to fixtures for unit tests."""
    return unit_fix_path


@pytest.fixture
def network():
    """Standard microcircuit network with testing analysis params."""
    network = lmt.models.Microcircuit(
        network_params=(config_path + 'network_params_microcircuit.yaml'),
        analysis_params=(config_path + 'analysis_params_test.yaml')
        )
    return network


@pytest.fixture
def microcircuit():
    """Standard microcircuit network with testing analysis params."""
    microcircuit = lmt.models.Microcircuit(
        network_params=(config_path + 'network_params_microcircuit.yaml'),
        analysis_params=(config_path + 'analysis_params_test.yaml')
        )
    return microcircuit


@pytest.fixture
def empty_network():
    """Network object with no parameters."""
    return lmt.models.Network()


@pytest.fixture
def network_dict_val_unit():
    """
    Simple example of all network dictionaries in val unit format in a dict.
    
    Returns:
    --------
    dict
        {network_params, analysis_params, results, results_hash_dict}
    """
    network_params = {'tau_m': {'val': 10, 'unit': 'ms'}}
    analysis_params = {'omegas': {'val': [1, 2, 3, 4], 'unit': 'hertz'}}
    results = {'test': {'val': 1, 'unit': 'hertz'}}
    results_hash_dict = {
        '46611a50a5da6eb3b7761b552bb28fc5': {
            'test': {'val': 1, 'unit': 'hertz'},
            'analysis_params': {'test_key': {'val': 1, 'unit': 'ms'}}
            }
        }
    return {'network_params': network_params,
            'analysis_params': analysis_params,
            'results': results,
            'results_hash_dict': results_hash_dict}
    
    
@pytest.fixture
def network_dict_quantity():
    """
    Simple example of all network dictionaries in quantity format in a dict.
    
    Returns:
    --------
    dict
        {network_params, analysis_params, results, results_hash_dict}
    """
    network_params = {'tau_m': 10 * ureg.ms}
    analysis_params = {'omegas': [1, 2, 3, 4] * ureg.Hz}
    results = {'test': 1 * ureg.Hz}
    results_hash_dict = {
        '46611a50a5da6eb3b7761b552bb28fc5': {
            'test': 1 * ureg.Hz,
            'analysis_params': {'test_key': 1 * ureg.ms}
            }
        }
    return {'network_params': network_params,
            'analysis_params': analysis_params,
            'results': results,
            'results_hash_dict': results_hash_dict}
    

@pytest.fixture
def std_results():
    """Standard microcircuit network results for testing analysis params."""
    results = load_val_unit_dict_from_h5(integration_fix_path
                                         + 'std_results.h5')
    return results['results']


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
    """Standard params from first two populations of microcircuit."""
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
                  nu_e_ext=np.array([1, 2]) * ureg.Hz,
                  nu_i_ext=np.array([1, 2]) * ureg.Hz,
                  populations=['23E', '23I'],
                  sigma=std_input,
                  sigma_set=np.array([6, 5]) * ureg.mV,
                  std_input=std_input,
                  tau_m=10 * ureg.ms,
                  tau_s=0.5 * ureg.ms,
                  tau_r=2.0 * ureg.ms,
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
def all_std_params_single_population():
    """Standard params from first population of microcircuit."""
    firing_rates = 0.71 * ureg.Hz
    mean_input = 3.30 * ureg.mV
    std_input = 6.19 * ureg.mV
    params = dict(C=250 * ureg.pF,
                  K=2199,
                  K_ext=1600,
                  N=20683,
                  V_th_abs=-50 * ureg.mV,
                  V_0_abs=-65 * ureg.mV,
                  d_e=1.5 * ureg.ms,
                  d_i=0.75 * ureg.ms,
                  d_e_sd=0.75 * ureg.ms,
                  d_i_sd=0.375 * ureg.ms,
                  delay_dist=None,
                  delay_dist_matrix=complex(1.000,
                                            -0.005) * ureg.dimensionless,
                  firing_rates=firing_rates,
                  g=4,
                  label='microcircuit',
                  matrix='MH',
                  mean_input=mean_input,
                  mu=mean_input,
                  mu_set=np.array([3]) * ureg.mV,
                  nu=firing_rates,
                  nu_ext=8 * ureg.Hz,
                  nu_e_ext=0 * ureg.Hz,
                  nu_i_ext=0 * ureg.Hz,
                  populations=['23E'],
                  sigma=std_input,
                  sigma_set=6 * ureg.mV,
                  std_input=std_input,
                  tau_m=10 * ureg.ms,
                  tau_s=0.5 * ureg.ms,
                  tau_r=2.0 * ureg.ms,
                  transfer_function=complex(0.653) * ureg.Hz / ureg.mV,
                  omega=20 * ureg.Hz,
                  omegas=20 * ureg.Hz,
                  w=87.8 * ureg.pA
                  )
    calc_dep_params(params)
    return params


@pytest.fixture
def all_std_params_single_population_unitless():
    """Standard params from first population of microcircuit."""
    firing_rates = 0.71
    mean_input = 3.30
    std_input = 6.19
    params = dict(C=250,
                  K=2199,
                  K_ext=1600,
                  N=20683,
                  V_th_abs=-50,
                  V_0_abs=-65,
                  d_e=1.5,
                  d_i=0.75,
                  d_e_sd=0.75,
                  d_i_sd=0.375,
                  delay_dist=None,
                  delay_dist_matrix=complex(1.000,
                                            -0.005),
                  firing_rates=firing_rates,
                  g=4,
                  label='microcircuit',
                  matrix='MH',
                  mean_input=mean_input,
                  mu=mean_input,
                  mu_set=np.array([3]),
                  nu=firing_rates,
                  nu_ext=8,
                  nu_e_ext=0,
                  nu_i_ext=0,
                  populations=['23E'],
                  sigma=std_input,
                  sigma_set=6,
                  std_input=std_input,
                  tau_m=10,
                  tau_s=0.5,
                  tau_r=2.0,
                  transfer_function=complex(0.653),
                  omega=20,
                  omegas=20,
                  w=87.8
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
def std_unitless_params(request, all_std_params):
    """
    Returns set of standard params needed by requesting tested function.
    
    For using this fixture, the function test class needs to have a class
    attribute `func`, which is the tested function as a staticmethod.
    """
    params = get_required_params(request.cls.func, all_std_params)
    _strip_units(params)
    return params


@pytest.fixture
def std_params_single_population(request, all_std_params_single_population):
    """
    Returns set of standard params needed by requesting tested function.
    
    For using this fixture, the function test class needs to have a class
    attribute `func`, which is the tested function as a staticmethod.
    """
    return get_required_params(request.cls.func,
                               all_std_params_single_population)


@pytest.fixture
def std_params_single_population_unitless(
        request, all_std_params_single_population):
    """
    Returns set of standard params needed by requesting tested function.
    
    For using this fixture, the function test class needs to have a class
    attribute `func`, which is the tested function as a staticmethod.
    """
    params = get_required_params(request.cls.func,
                                 all_std_params_single_population)
    for key, value in params.items():
        try:
            params[key] = value.magnitude
        except AttributeError:
            pass
    return params


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
    other as a parametrization.
    """
    # check if requesting test class has class attribute func
    if hasattr(metafunc.cls, 'func'):
        func = metafunc.cls.func
    # if it does not, just return and don't parametrize
    else:
        return None
    
    if "pos_keys" in metafunc.fixturenames:
        pos_keys = get_required_keys(func, all_pos_keys)
        # define parametrization
        metafunc.parametrize("pos_keys", pos_keys)

    elif "output_test_fixtures" in metafunc.fixturenames:
        # list of input arguments for the tested function for each regime
        params = [get_required_params(func, dict(results, **params))
                  for params, results in zip(all_params, results)]
        # list of outputs for the tested function for each regime
        output = get_output_for_keys_of_metafunc(metafunc, results, params)
        fixtures = [dict(output=output, params=params) for output, params
                    in zip(output, params)]
        metafunc.parametrize("output_test_fixtures", fixtures,
                             ids=ids_all_regimes)
        
    elif "output_fixtures_noise_driven" in metafunc.fixturenames:
        # list of input arguments for the tested function for each regime
        params = [get_required_params(func, dict(results[0], **all_params[0]))]
        # list of outputs for the tested function for each regime
        output = get_output_for_keys_of_metafunc(metafunc, results[0:1],
                                                 params)
        fixtures = [dict(output=output, params=params) for output, params
                    in zip(output, params)]
        metafunc.parametrize("output_fixtures_noise_driven", fixtures,
                             ids=ids_all_regimes[0:1])
        
    elif "output_fixtures_mean_driven" in metafunc.fixturenames:
        # list of input arguments for the tested function for each regime
        params = [get_required_params(func, dict(results[1], **all_params[1]))]
        # list of outputs for the tested function for each regime
        output = get_output_for_keys_of_metafunc(metafunc, results[1:],
                                                 params)
        fixtures = [dict(output=output, params=params) for output, params
                    in zip(output, params)]
        metafunc.parametrize("output_fixtures_mean_driven", fixtures,
                             ids=ids_all_regimes[1:])

    elif "unit_fixtures" in metafunc.fixturenames:
        file = metafunc.cls.fixtures
        fixtures = h5.load(unit_fix_path + file)
        ids = sorted(fixtures.keys())
        fixture_list = [dict(output=fixtures[id]['output'],
                        params=fixtures[id]['params'])
                        for id in ids]
        # import pdb; pdb.set_trace()
        metafunc.parametrize("unit_fixtures", fixture_list, ids=ids)

    elif "unit_fixtures_fully_vectorized" in metafunc.fixturenames:
        file = metafunc.cls.fixtures
        fixtures = h5.load(unit_fix_path + file)
        ids = sorted(fixtures.keys())
        fixture_list = [dict(output=fixtures[id]['output'],
                        params=fixtures[id]['params'])
                        for id in ids if id == 'fully_vectorized']
        # import pdb; pdb.set_trace()
        metafunc.parametrize("unit_fixtures_fully_vectorized", fixture_list,
                             ids=['fully_vectorized'])
