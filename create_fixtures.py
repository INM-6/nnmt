import numpy as np
import yaml

import lif_meanfield_tools as lmt
ureg = lmt.ureg

case = 2
if case == 0:
    parameters = 'examples/network_params_microcircuit.yaml'
    regime = 'noise_driven'
elif case ==1:
    parameters = 'examples/minimal_negative.yaml'
    regime = 'negative_firing_rate'
elif case == 2:
    parameters = 'examples/small_network.yaml'
    regime = 'mean_driven'
else: 
    print('Case not defined! Choose existing case, otherwise nothing happens!')

network = lmt.Network(parameters, 'examples/analysis_params.yaml')

network.working_point()

params = network.network_params

fixtures = dict(params, **network.results)

fixtures = lmt.input_output.quantities_to_val_unit(fixtures)

for k,v in fixtures.items():
    try:
        fixtures[k] = v.tolist()
    except AttributeError:
        try: 
            fixtures[k]['val'] = v['val'].tolist()
        except AttributeError:
            pass
        except TypeError:
            pass
        
with open('tests/unit/fixtures/{}_regime.yaml'.format(regime), 'w') as file:
    yaml.dump(fixtures, file)
    