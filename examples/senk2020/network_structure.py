"""
Structure of ring network
=========================

This example illustrates the structure of the ring network used by
:cite:t:`senk2020`.
Mean-field results of this model are shown in the examples
:doc:`fit_transfer_function` and :doc:`spatial_patterns`.

"""

##########################################################################
# Here, we use the Python package ``svgutils`` to integrate an externally
# generated network sketch into the figure.
# This package is not part of the default Python environment of NNMT.
# In the end of the script, we use the external tool ``inkscape`` via the
# command line to convert from ``svg`` format to ``eps``.

import os
import sys
from nnmt.models.basic import Basic as BasicNetwork
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.style.use('frontiers.mplstyle')
mpl.rcParams.update({'legend.fontsize': 'medium',  # old: 5.0 was too small
                     'axes.titlepad': 0.0})
try:
    import svgutils.transform as sg
except BaseException:
    pass

assert 'svgutils' in sys.modules, (
    'This figure requires: "import svgutils.transform as sg"')

##########################################################################
# First, we define plotting parameters.

params = {
    # figure file name (without extension)
    'figure_fn': 'Senk2020_network_structure',

    # file name of external sketch
    'sketch_fn': 'Senk2020_sketch.svg',

    # figure width in inch
    'figwidth_1col': 85. / 25.4,

    # label and corresponding scaling parameter for plotted quantities
    'quantities': {
        'displacement': {
            'label': 'displacement $d$ (mm)',
            'scale': 1e3}},

    # colors for excitation and inhibition
    'colors': {
        'ex_blue': '#4C72B0',
        'inh_red': '#C44E52'}}

###############################################################################
# We also define a helper function for adding labels to figure panels.


def _add_label(ax, label, xshift=0., yshift=0., scale_fs=1.):
    """
    Adds label to plot panel given by axis.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes object
        Axes.
    label : str
        Letter.
    xshift : float
        x-shift of label position.
    yshift : float
        y-shift of label position.
    scale_fs : float
        Scale factor for font size.
    """
    label_pos = [0., 1.]
    ax.text(label_pos[0] + xshift, label_pos[1] + yshift, '(' + label + ')',
            ha='left', va='bottom',
            transform=ax.transAxes, fontweight='bold',
            fontsize=mpl.rcParams['font.size'] * scale_fs)

##########################################################################
# The figure to illustrate the network structure spans one column.
# It is divided into three panels.
# Panel A contains the external network sketch which we will add in after
# the other panels will be completed.


fig = plt.figure(figsize=(params['figwidth_1col'], params['figwidth_1col']))
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0)

# panel for external network sketch
_add_label(plt.subplot(gs[0, :]), 'A', xshift=-0.04, yshift=-0.1)
plt.gca().set_axis_off()

##########################################################################
# In panel B, we illustrate the ring-like network structure.

ax = plt.subplot(gs[1, 0])
_add_label(ax, 'B', xshift=-0.085)

circ = Circle(xy=(0, 0), radius=1, color='k', fill=False)
ax.add_artist(circ)

blue = params['colors']['ex_blue']
red = params['colors']['inh_red']

n_in = 10
ex2in = 4

step = 2 * np.pi / n_in

offset = 0
for i in np.arange(1 + ex2in):
    if i == 0:  # inh.
        marker = 'o'
        color = red
    else:  # exc.
        marker = '^'
        color = blue

    xy = []
    for i in np.arange(n_in):
        x = (1 + offset) * np.cos(i * step)
        y = (1 + offset) * np.sin(i * step)
        xy.append([x, y])
    xy = np.array(xy)

    ax.plot(xy[:, 0], xy[:, 1],
            marker=marker, markersize=mpl.rcParams['lines.markersize'],
            linestyle='', color=color)
    offset += 0.2

ax.annotate('ring\n network', (0, 0), va='center', ha='center')

lim = 2
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect('equal')

plt.axis('off')

##########################################################################
# In panel C, we illustrate the boxcar-shaped spatial connectivity profile.


def _get_p(rs, width):
    p = np.zeros(len(rs))
    # p is normalized to 1
    height = 1. / (2 * width)
    p[np.where(np.abs(rs) <= width)] = height
    return p


gs_prof = gs[1, 1].subgridspec(4, 1)
ax = plt.subplot(gs_prof[1:3])
_add_label(ax, 'C', yshift=1.08)

blue = params['colors']['ex_blue']
red = params['colors']['inh_red']

network = BasicNetwork(
    network_params='Senk2020_network_params.yaml',
    analysis_params='Senk2020_analysis_params.yaml')
ewidth = network.network_params['width'][0] * \
    params['quantities']['displacement']['scale']
iwidth = network.network_params['width'][1] * \
    params['quantities']['displacement']['scale']

max_x = np.max([ewidth, iwidth])
rs = np.arange(-1.5 * max_x, 1.5 * max_x, 1e-5)  # in mm

ep = _get_p(rs, ewidth)
ip = _get_p(rs, iwidth)
ax.plot(rs, ep, blue)
ax.plot(rs, ip, red)

xstp = ewidth / 6.
ax.annotate('E',
            [ewidth + xstp, 1. / (2. * ewidth)],
            color=blue,
            va='top', ha='left')

ax.annotate('I',
            [iwidth + xstp, 1. / (2. * iwidth)],
            color=red,
            va='top', ha='left')

ax.set_xlim(rs[0], rs[-1])
ax.set_xlabel(params['quantities']['displacement']['label'])
ax.get_yaxis().set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_title('connection\n probability $p$')

##########################################################################
# Finally, the external network sketch is included and the figure is saved to
# file.

svg_mpl = sg.from_mpl(fig, savefig_kw=dict(transparent=True))
w_svg, h_svg = svg_mpl.get_size()
svg_mpl.set_size((w_svg + 'pt', h_svg + 'pt'))
svg_sketch = sg.fromfile(params['sketch_fn']).getroot()
svg_sketch.moveto(x=50, y=10, scale_x=1.5)
svg_mpl.append(svg_sketch)
svg_mpl.save(f"{params['figure_fn']}.svg")
os_return = os.system(
    f"inkscape --export-eps={params['figure_fn']}.eps {params['figure_fn']}.svg")
if os_return == 0:
    os.remove(f"{params['figure_fn']}.svg")
else:
    print('Conversion to eps using inkscape failed, keeping svg.')
