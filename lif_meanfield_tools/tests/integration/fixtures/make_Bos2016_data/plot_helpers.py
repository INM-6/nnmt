import numpy as np
from matplotlib.pylab import *
import meanfield.circuit as circuit

populations = ['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I']
layers = ['L23', 'L4', 'L5', 'L6']

# returnes parameter set used in manuscript
def get_parameter_microcircuit():
    # factor for standard deviation of delay distribution
    dsd = 1.0
    circ = circuit.Circuit('microcircuit', analysis_type=None)
    I_new = circ.I.copy()
    # reduce indegrees from 4I to 4E
    I_new[2][3] = np.round(circ.I[2][3]*(1.0-0.15))
    Next_new = circ.Next.copy()
    # adjust external input to 4E 
    Next_new[2] -= 300
    Next_new[2] *= (1.0-0.011)
    new_params = {'de_sd': circ.de*dsd, 'di_sd': circ.di*dsd,
                  'I': I_new, 'Next': Next_new, 
                  'delay_dist': 'truncated_gaussian'}
    params = circ.params
    params.update(new_params)
    return params

def set_fig_defaults():
    # resolution of figures in dpi
    rcParams['figure.dpi'] = 300
    rcParams['legend.isaxes'] = False
    rcParams['figure.subplot.left'] = 0.1
    rcParams['figure.subplot.right'] = 0.95
    rcParams['figure.subplot.top'] = 0.8
    rcParams['figure.subplot.bottom'] = 0.15
    # size of markers (points in point plots)
    rcParams['lines.markersize'] = 3.0
    rcParams['text.usetex'] = False
    rcParams['ps.useafm'] = False   # use of afm fonts, results in small files
    rcParams['ps.fonttype'] = 3    # Output Type 3 (Type3) or Type 42 (TrueType)

def reorder_matrix(M):
    M_reordered = np.zeros((8,8))
    for i in range(8):
        M_reordered[i] = M[8-1-i]
    return M_reordered

def get_parameter_plot():
    colors = [[] for i in range(4)]
    colors[1] = [(0.0, 0.7, 0.0), (0.0, 1.0, 0.0)]
    colors[3] = [(0.0, 0.0, 0.4), (0.0, 0.0, 1.0)]
    colors[0] = [(0.7, 0.0, 0.0), (1.0, 0.0, 0.0)]
    colors[2] = [(0.5, 0.0, 0.5), (1.0, 0.0, 1.0)]
    return colors
