LIF Mean-field Tools
====================
This python package provides useful tools for analysing neuronal networks
consisting of leaky integrate and fire (LIF) neurons. These tools are based on
mean-field theory of neuronal networks. That is why this package is called
lif_meanfield_tools (LMT). 

Using this package, you can easily calculate quantities like firing rates, power
spectra, and many more, which give you a deeper and more intuitive understanding
of what your network does. If your network is not behaving the way you want it
to, these tools might help you to figure out, or even tell you, what you need to
change in order to achieve the desired behaviour. 

We are always trying to improve and simplify the usage of this package. Therefore, 
it is easy to store (and in the future, to plot) your results and reuse them for 
further analyses. We are always happy about feedback. So please do not hesitate
to contact us, if you think that we could improve you life (or workflow). 

# How to get started / Installation
Not sure.

Install h5py wrapper:
pip install git+https://github.com/INM-6/h5py_wrapper.git

Install pint
pip install pint

Install lif_meanfield_tools:
pip install git+https://github.com/INM-6/lif_meanfield_tools.git

python3 setup.py install

# How to use this package
In order to give you a quick and simple start, we wrote a little example script, 
which you can find on github. First of all, you should have a look at this file. 
Actually, we hope that the usage might be self-explanatory, once you have seen
an example. But, if you need a little more hints, just continue reading.

For using LMT, you need to store all your network parameters and your analysis
paramters in yaml files, as we have done it for the example script. If you don't
know how the yaml file format works, you could either first read something 
about it, or use our example yaml files as templates. 

So, let us start coding. First of all you need to import the package itself. 
Additionally, you might want to define a variable for the pint unit registry (ureg). 
This is needed for dealing with units and some of  the functionality implemented,
needs the usage of pint units. 

Now, you can instantiate a network, by calling the central LMT class 'Network' and 
passing the yaml file names. A Network object represents your network. When it is 
instantiated, it first calculates all the parameters that are derived from the 
passed parameters. Then, it stores all the parameters associated with the network 
under consideration. Additionally, it checks whether this parameters have been used
for an analysis before, and if so loads the corresponding results. Newly calculated 
results are stored withing the Network object as well. 

A Network object has the ability to tell you about it's properties, simply by calling
the corresponding method as

	network.property()

Here, 'property' can be replaced by lots of stuff, like for example, 'firing_rates', 
'transfer_function', or 'power_spectra'. You can find the complete list of Network
methods at the end of this chapter. When such a method is called, the network first checks, 
whether this quantity has been calculated before. If so, it returns the stored value. 
If not, it does the calculations, stores the results, and returns them.

Sometimes, you might want to know a property for some specific parameter, like for 
example the power_spectra at a certain frequency. Then, you need to pass the parameter
including it's unit to the method, e.g.
	
	network.property(10 * ureg.Hz)

If you want to save your results, you can simply call 

	network.save()

and the calculated results, together with the corresponding paramters, will be stored
inside a h5 file, whose name contains a hash, which reflects the used network parameters. 

Network methods:
save
show
change_parameters
firing_rates
mean_input
std_input
working_point
delay_dist_matrix
transfer_function
sensitivity_measure
power_spectra
eigenvalue_spectra
r_eigenvec_spectra
l_eigenvec_spectra

A detailed explanation of these methods follows shortly.

# History of this Project
Mean-field theory is a very handy tool, when you want to understand the behaviour of 
your network. Using this theory allows you to predict some features of a network, without
running a single (often very time consuming) simulation. 

At our institute, the INM-6 at the Research Center Juelich, we, among other things, investigate 
and develop such mean-field tools. Over the years, more and more of the tools, developed by 
ourselves and other researchers, have been implemented. In particular, this primary work has 
been done by Hannah Bos, Jannis Schuecker and Moritz Helias. The corresponding publications 
can be found at the end of this chapter. But, this code never was intended for a wider user
base. For this reason, the code was a little cumbersome and the usage was not very intuitive. 

However, we wanted to use the convenient tools, that were kind of concealed by the 
complexity of the code as it was at that time. Hence, we decided to restructure and rewrite
the code in such a way, that people could use it without understanding the underlying theory. 
We changed a lot of things. We simplified the code. We introduced units and decided to store 
the parameters in seperate yaml files. We wanted the user to only have to interact with one 
module. So, we collected all the functionality in the network.py module. We ported the code 
to python 3. We made the whole thing a package. We expanded the documentation a lot. We 
simplified saving results together with the parameters. And so on. 

What we ended up with is the package that you are currently interested in. It contains 
several tools for analyzing neuronal networks. And it is very simple to use. 

# Structure
[structure plot] 

lif_meanfield_tools consists of four modules:

The central module is network.py. It defines a class 'Network' which is a container for network 
parameters, analysis parameters and calculated results. Network comes with all the methods that 
can be used to calculate network properties, like for example firing rates or power spectra. 
Additionally, there are some 'administrative' methods for changing parameters or saving. 

input_output.py is called by network.py for everything that is related to input or output. 
Here we defined saving and loading routines, quantity format conversions and hash creation.

meanfield_calcs.py is the module which is called everytime a method of Network is called. Here
we put all the mathematical details of the mean-field theory. 

aux_calcs.py is a module where auxiliary calculations that are needed in meanfield_calcs.py are 
defined. It is difficult to draw a line between the calculations that belong to menafield_calcs 
and the ones that belong to aux_calcs. We mainly introduced this module to be able to keep 
as much of the former code's structure as possible. 



