import numpy as np
import glob

path_base = './data/'

def get_gids(folder):
    path = path_base + folder
    data = np.loadtxt(path + '/population_GIDs.dat')
    gid_mins = np.transpose(data)[0].astype(int)
    pop_sizes = np.transpose(data)[1].astype(int)-gid_mins+1
    return gid_mins, pop_sizes

def cut_arrays(x, y, xmin, xmax):
    y = y[np.where(x<xmax)]
    x = x[np.where(x<xmax)]
    y = y[np.where(x>=xmin)]
    x = x[np.where(x>=xmin)]
    return x, y

def read_one_pop(folder, pop, T, tmin, gid_lims=None):
    path = path_base + folder
    gid_mins, pop_sizes = get_gids(folder)
    nr_gids = pop_sizes[pop]
    gid_min = gid_mins[pop]
    gid_max = gid_mins[pop] + nr_gids - 1
    if gid_lims is not None:
        gid_min = gid_min + gid_lims[0]
        gid_max = gid_min + gid_lims[1]
    spikes = [[] for i in range(nr_gids)]
    filestart = path + '/spikes_' + str(pop/2) + '_' + str(pop%2) + '*'
    filelist = glob.glob(filestart)
    for filename in filelist:
        input_file = open(filename,'r')
        for index,line in enumerate(input_file):
            data = map(float, line.split())
            if int(data[0])>=gid_min and int(data[0])<gid_max:
                if data[1]>tmin and data[1]<T:
                    spikes[int(data[0])-gid_min].append(data[1])
        input_file.close()
    return spikes

def read_all_neurons(folder, pop, T, tmin, gid_lims=None):
    gid_mins, pop_sizes = get_gids(folder)    
    t_temp = 0
    path = path_base + folder
    nr_gids = pop_sizes[pop]
    gid_min = gid_mins[pop]
    gid_max = gid_mins[pop] + nr_gids - 1
    if gid_lims is not None:
        gid_min = gid_min + gid_lims[0]
        gid_max = gid_min + gid_lims[1]
    spikes = []
    filestart = path + '/spikes_' + str(pop/2) + '_' + str(pop%2) + '*'
    filelist = glob.glob(filestart)
    for filename in filelist:
        input_file = open(filename,'r')
        for index,line in enumerate(input_file):
            data = map(float, line.split())
            if int(data[0])>=gid_min and int(data[0])<gid_max:
                try:
                    if data[1]>tmin and data[1]<T:
                        spikes.append(data[1])
                    t_temp = data[1]
                except:
                    continue
        input_file.close()
    return spikes

def get_mu_rate(folder, pop, T, tmin):
    gid_mins, pop_sizes = get_gids(folder)
    all_spike_times = read_one_pop(folder, pop, T, tmin)
    all_rates = [len(spike_times)/(T-tmin) for spike_times in all_spike_times]
    rate_av = sum(all_rates)/float(pop_sizes[pop])
    return rate_av

# gids limit is specified as an intervall, [a,b], returning data for the 
# gids a+gid_min up to b+gid_min
def get_data_dotplot(folder, pop, tmin, tmax, gid_lims=None):
    gid_mins, pop_sizes = get_gids(folder)
    path = path_base + folder
    nr_gids = pop_sizes[pop]
    gid_min = gid_mins[pop]
    gid_max = gid_mins[pop] + nr_gids - 1
    if gid_lims is not None:
        gid_min = gid_min + gid_lims[0]
        gid_max = gid_min + gid_lims[1]
    spikes = []
    gids = []
    filestart = path + '/spikes_' + str(pop/2) + '_' + str(pop%2) + '*'
    filelist = glob.glob(filestart)
    for filename in filelist:
        input_file = open(filename,'r')
        for index,line in enumerate(input_file):
            data = map(float, line.split())
            if int(data[0])>=gid_min and int(data[0])<gid_max:
                if data[1]>=tmin and data[1]<=tmax:
                    spikes.append(data[1])
                    gids.append(data[0])
        input_file.close()
    return spikes, gids, gid_min, gid_max

# creates spectrum and returns c =p^2/T, p is the absolute value of the 
# result of an fft
# returns frequencies, power and rate
# power and rate are normalised by population size if pop_size is specified
# if mode is specified as 'fluctuations' the means is substraced, 
# this influences the spectrum only at f=0
def create_spec(spike_times, T, tmin,  pop_size=1, mode=None, 
                bin_size=1.0):
    rate = len(spike_times)/(T-tmin)*1/float(pop_size)
    bins, hist = instantaneous_spike_count([spike_times], bin_size, 
                                           tmin, tmax=T) 
    # normalise hist according to population size
    hist = [hist[0]/float(pop_size)]
    # remove expected number of spikes from each bin
    if mode == 'fluctuations':
        exp_spikes = rate*bin_size
        hist = [hist[0]-exp_spikes]
    freq, p = powerspec(hist, bin_size) 
    return freq, p[0], rate

def get_spec(folder, pop, T, tmin, fmax=100., gid_lims=None, 
             bin_size=1.0):
    gid_mins, pop_sizes = get_gids(folder)
    N = float(pop_sizes[pop])
    gid_min = gid_mins[pop]
    if gid_lims is not None:
        gid_min = gid_min + gid_lims[0]
        gid_max = gid_min + gid_lims[1]
        N = float(gid_max - gid_min)
    spike_times = read_all_neurons(folder, pop, T, tmin, gid_lims)
    # inserting pop_size here is equivalent to dividing power by 
    # pop_size^2 afterwards
    freq, p, rate = create_spec(spike_times, T, tmin, N, bin_size=bin_size)
    freq, p = cut_arrays(freq, p, 0.1, fmax)
    return freq, p

def powerspec(data, tbin, Df=None, units=False, N=None):
    '''
    Calculate (smoothed) power spectra of all timeseries in data. 
    If units=True, power spectra are averaged across units.
    Note that averaging is done on power spectra rather than data.

    Power spectra are normalized by the length T of the time series -> no scaling with T. 
    For a Poisson process this yields:

    **Args**:
       data: numpy.ndarray; 1st axis unit, 2nd axis time
       tbin: float; binsize in ms
       Df: float/None; window width of sliding rectangular filter (smoothing), None -> no smoothing
       units: bool; average power spectrum 

    **Return**:
       (freq, POW): tuple
       freq: numpy.ndarray; frequencies
       POW: if units=False: 2 dim numpy.ndarray; 1st axis unit, 2nd axis frequency
            if units=True:  1 dim numpy.ndarray; frequency series

    **Examples**:
       >>> powerspec(np.array([analog_sig1,analog_sig2]),tbin, Df=Df)
       Out[1]: (freq,POW)
       >>> POW.shape
       Out[2]: (2,len(analog_sig1))

       >>> powerspec(np.array([analog_sig1,analog_sig2]),tbin, Df=Df, units=True)
       Out[1]: (freq,POW)
       >>> POW.shape
       Out[2]: (len(analog_sig1),)

    '''
    if N is None:
        N = len(data)
    freq, DATA = calculate_fft(data, tbin)
    df = freq[1] - freq[0]
    T = tbin * len(freq)
    POW = np.power(np.abs(DATA),2)
    if Df is not None:
        POW = [movav(x, Df, df) for x in POW]
        cut = int(Df / df)
        freq = freq[cut:]
        POW = np.array([x[cut:] for x in POW])
        POW = np.abs(POW)
    assert(len(freq) == len(POW[0]))
    if units is True:
        POW = 1./N*np.sum(POW, axis=0)
        assert(len(freq) == len(POW))
    POW *= 1. / T * 1e3  # normalization, power independent of T
    return freq, POW

def calculate_fft(data, tbin):
    '''
    calculate the fouriertransform of data
    [tbin] = ms
    '''
    if len(np.shape(data)) > 1:
        n = len(data[0])
        return np.fft.fftfreq(n, tbin * 1e-3), np.fft.fft(data, axis=1)
    else:
        n = len(data)
        return np.fft.fftfreq(n, tbin * 1e-3), np.fft.fft(data)

def movav(y, Dx, dx):
    '''
    calculate average of signal y by using sliding rectangular
    window of size Dx using binsize dx
    '''
    if Dx <= dx:
        return y
    else:
        ly = len(y)
        r = np.zeros(ly)
        n = np.int(np.round((Dx / dx)))
        r[0:np.int(n / 2.)] = 1.0 / n
        r[-np.int(n / 2.)::] = 1.0 / n
        R = np.fft.fft(r)
        Y = np.fft.fft(y)
        yf = np.fft.ifft(Y * R)
        return yf

def instantaneous_spike_count(data, tbin, tmin=None, tmax=None):
    '''
    Create a histogram of spike trains
    returns bins, hist
    '''
    if tmin is None:
        tmin = np.min([np.min(x) for x in data if len(x) > 0])
    if tmax is None:
        tmax = np.max([np.max(x) for x in data if len(x) > 0])
    assert(tmin < tmax)
    bins = np.arange(tmin, tmax + tbin, tbin)
    hist = np.array([np.histogram(x, bins)[0] for x in data])
    return bins[:-1], hist

def get_inst_rate(folder, pop, T, tmin, gid_lims=None, bin_size=1.0):
    gid_mins, pop_sizes = get_gids(folder)
    N = float(pop_sizes[pop])
    gid_min = gid_mins[pop]
    if gid_lims is not None:
        gid_min = gid_min + gid_lims[0]
        gid_max = gid_min + gid_lims[1]
        N = float(gid_max - gid_min)
    spike_times = read_all_neurons(folder, pop, T, tmin, gid_lims)
    bins, hist = instantaneous_spike_count([spike_times], bin_size, 
                                           tmin, tmax=T) 
    rates = hist/N*1000./bin_size
    return bins, rates
