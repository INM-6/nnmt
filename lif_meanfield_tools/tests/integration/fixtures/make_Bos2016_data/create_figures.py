import pyx
import microcircuit
import eigenmodes
import eigenvalue_trajectories
import sensitivity_measure
import Z_validation
import oscillation_origin

# Before execution please follow the installation instructions in 
# meanfield/README.txt

# simulation data is analysed or results are read from file
# if set to True existing data in results.h5 will be overwritten
calcData = False
# meanfield results are calculated or results are read from file
# if set to True existing data in results.h5 will be overwritten
calcAna = False
# meanfield results including firing rates and transfer function are
# calculated (implies calcAna=True)
# if set to True existing data in results.h5 and results_microcircuit.h5
# will be overwritten
calcAnaAll = False

# ### Figure 1 ###
print 'Create figure 1.'
microcircuit.plot_fig(calcData, calcAna, calcAnaAll)
c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(0, 0, "microcircuit.eps"))
c.insert(pyx.epsfile.epsfile(0.0, 8.0, "column.eps", scale=0.62))
c.writeEPSfile("Fig1.eps")

# # ### Figure 3 ###
print 'Create figure 3.'
eigenmodes.plot_fig(calcAna, calcAnaAll)
c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(0, 0, "eigenmodes.eps"))
c.insert(pyx.epsfile.epsfile(1.0, 4.5, "eigenmodes_sketch.eps", scale=0.3))
c.writeEPSfile("Fig3.eps")

# ### Figure 4 ###
print 'Create figure 4.'
eigenvalue_trajectories.plot_fig(calcAna, calcAnaAll, 'Fig4.eps')

# ### Figure 5 ###
print 'Create figure 5.'
sensitivity_measure.plot_fig_gamma(calcAna, calcAnaAll, 'Fig5.eps')

# ### Figure 6 ###
print 'Create figure 6.'
sensitivity_measure.plot_fig_high_gamma(calcAna, calcAnaAll, 'Fig6.eps')

# ### Figure 7 ###
print 'Create figure 7.'
sensitivity_measure.plot_fig_low(calcAna, calcAnaAll, 'Fig7.eps')

# ### Figure 8 ###
print 'Create figure 8.'
Z_validation.plot_fig(calcData, calcAna, calcAnaAll, 'Fig8.eps')

# ### Figure 9 ###
print 'Create figure 9.'
oscillation_origin.plot_fig(calcData, calcAna, calcAnaAll, 'Fig9.eps')
