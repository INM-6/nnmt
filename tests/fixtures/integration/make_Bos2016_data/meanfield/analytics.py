"""analytics.py: Class in which all static and dynamical properties of 
the circuit are calculated.

Authors: Hannah Bos, Jannis Schuecker
"""

import numpy as np
from scipy.special import erf
import os
import h5py_wrapper.wrapper as h5
import hashlib as hl
import siegert
import transfer_function as tf

class Analytics(object):

    def __init__(self):
        pass

    def update_variables(self, dict):
        """Converts all keys of the inserted dictionary and the parameter
        dictionary into class variables.
        """
        for key, value in dict.items():
            setattr(self, key, value)
        if 'params' in dict:
            for key, value in dict['params'].items():
                setattr(self, key, value)
            # include parameter from dictionary params into dictionary dict
            dict.update(dict['params'])
        if 'populations' in dict:
            self.dimension = len(self.populations)
        if 'th_rates' in dict:
            self.D = np.diag(np.ones(self.dimension))*self.th_rates/self.N

    def add_to_file(self, dic):
        '''Adds data to h5 file, with data and path specified in dict.'''
        self.file_base = os.getcwd()+'/'+self.params['datafile']
        h5.add_to_h5(self.file_base, {self.param_hash: dic},
                     'a', overwrite_dataset=True)

    def read_from_file(self, path):
        '''Reads data from h5 file at location path.'''
        self.file_base = os.getcwd()+'/'+self.params['datafile']
        path_base = '/' + self.param_hash + '/'
        data = h5.load_h5(self.file_base, path_base + path)
        return data

    def create_hash(self, dic):
        """Returns hash from elements in dic."""
        label = ''
        for key,value in dic.items():
            if isinstance(value, (np.ndarray, np.generic) ):
                label += value.tostring()
            else:
                label += str(value)
        return hl.md5(label).hexdigest()

    def get_mean(self, f):
        '''Returns vector of mean inputs to populations depending on
        the firing rates f of the populations.
        Units as in Fourcoud & Brunel 2002, where current and potential
        have the same units, unit of output array: pA*Hz
        '''
        # contribution from within the network
        m0 = np.dot(self.I*self.W, f)
        # contribution from external sources
        m_ext = self.w*self.Next*self.v_ext
        m = m0+m_ext
        return m

    def get_variance(self, f):
        '''Returns vector of variances of inputs to populations depending
        on the firing rates f of the populations.
        Units as in Fourcoud & Brunel 2002, where current and potential
        have the same units, unit of output array: pA*Hz
        '''
        # contribution from within the network
        var0 = np.dot(self.I*self.W*self.W, f)
        # contribution from external sources
        var_ext = self.w*self.w*self.Next*self.v_ext
        var = var0+var_ext
        return var

    def firing_rates_integration(self):
        '''Returns vector of population firing rates in Hz.'''
        print 'Calculate firing rates.'
        # conversion factor for currents to voltages # JS: In the conversion factor tau_s should also appear, but I think it is absorbed in the weights, right? Should we just take out the comment here?
        fac = 1 / (self.params['C'])
        # conversion from ms to s
        taum = self.params['taum'] * 1e-3
        tauf = self.params['tauf'] * 1e-3
        taur = self.params['taur'] * 1e-3
        dV =  self.params['Vth'] - self.params['V0']
        rate_function = lambda mu, sigma: siegert.nu0_fb433(
            taum, tauf, taur, dV, 0., mu, sigma)

        def get_rate_difference(rates):
            mu = self.get_mean(rates)
            sigma = np.sqrt(self.get_variance(rates))
            new_rates = np.array(map(rate_function, taum*fac*mu,
                                     np.sqrt(taum)*fac*sigma))
            return -rates + new_rates

        dt = 0.05
        y = np.zeros((2, self.dimension))
        eps = 1.0
        while eps >= 1e-5:
            delta_y = get_rate_difference(y[0])
            y[1] = y[0] + delta_y*dt
            epsilon = (y[1] - y[0])
            eps = max(np.abs(epsilon))
            y[0] = y[1]

        return y[1]

    def create_firing_rates(self):
        '''Returns vector of population firing rates in Hz.'''
        if self.from_file == True:
            try:
                path = 'firing_rates'
                rates = self.read_from_file(path)
                print 'Read firing rates from file.'
            except:
                rates = self.firing_rates_integration()
        else:
            rates = self.firing_rates_integration()
        if self.to_file == True:
            dic = {'firing_rates': rates}
            self.add_to_file(dic)
        return rates

    def create_H_df(self):
        '''Returns matrix of instantaneous rate jumps, such that
        H(omega=0)/H_df = 1.
        '''
        H_df = self.create_H(0.0)
        self.H_df = H_df
        return H_df

    def create_H_df(self, params=None, mode=None):
        """Returns matrix of instantaneous rate jumps, such that
        H(omega=0)/H_df = 1.

        Keyword arguments:
        params: dictionary specifying parameter of the transfer function
                when approximated by an exponential (needs to be
                specified when tf_mode = 'empirical')
        mode: string specifying by which method the transfer function
              is calculated (needs to be specified if the function is
              called before tf_mode was set as class variable)
        """
        if mode is not None:
            mode = self.tf_mode
        if mode == 'empirical':
            Hdf = np.zeros((self.dimension,self.dimension))
            tau_I = params['tau_impulse_p']*0.001
            tau_E = params['tau_impulse_n']*0.001
            Hdf[0:self.dimension+1:2] = params['delta_f_p']*tau_I
            Hdf[1:self.dimension+1:2] = params['delta_f_n']*tau_E
        elif mode == 'analytical':
            H_df = self.create_H(0.0)
        return H_df

    def mparray2npcomplex(self, a):
        """Converts a NumPy array of mpmath objects (mpf/mpc)
        to a NumPy array of complex.
        Code from https://code.google.com/p/mpmath/issues/attachmentText?
        id=61&aid=-1647199830253661025&name=mpmath_utils.py&token=
        d898b42bcbaa85abce505d3423b0ca88
        """
        _cfunc_complex = np.vectorize(complex)
        tmp = a.ravel()
        res = _cfunc_complex(tmp)
        return res.reshape(a.shape)

    def calc_transfer_function(self, mu, sigma):
        """Returns transfer function for one mean and variance."""
        print 'Calculate transfer function.'
        tf_one_pop = np.array([
            tf.transfer_function(
                w, self.params, mu, sigma) for w in self.omegas])
        return tf_one_pop

    def create_transfer_function(self):
        """Returns transfer functions for all populations."""
        trans_func = []
        for i in range(self.dimension):
            mu = self.mu[i]*self.taum*1e-3/self.C
            sigma = np.sqrt(self.var[i]*self.taum*1e-3)/self.C
            label = str(mu)+str(sigma)
            label += str(self.fmin)+str(self.fmax)+str(self.df)

            if self.from_file == True:
                try:
                    path = 'transfer_function/' + label
                    tf_one_pop = self.read_from_file(path)
                    print 'Read transfer function from file.'
                except KeyError:
                    tf_one_pop = self.calc_transfer_function(mu, sigma)
            else:
                tf_one_pop = self.calc_transfer_function(mu, sigma)
            # convert arrays of mpc objects to numpy arrays
            tf_one_pop = self.mparray2npcomplex(tf_one_pop)
            if self.to_file == True:
                if np.array(tf_one_pop).dtype.name == 'object':
                    tf_one_pop = [complex(n.real,n.imag)
                                       for n in tf_one_pop]
                dic = {'transfer_function':{label: tf_one_pop}}
                self.add_to_file(dic)

            trans_func.append(tf_one_pop)
        return trans_func

    def Phi(self, x):
        '''Helper function for Gaussian delay distribution.'''
        return 0.5*(1.0+erf(x/np.sqrt(2)))

    def create_delay_dist_matrix(self, omega):
        '''Returns matrix of delay distribution specific pre-factors at
        frequency omega.
        Assumes lower boundary for truncated Gaussian distributed delays
        to be zero (exact would be dt, the minimal time step).
        '''
        mu = self.Delay * 0.001
        sigma = self.Delay_sd * 0.001
        D = np.ones((self.dimension,self.dimension))

        if self.delay_dist == 'none':
            return D*np.exp(-complex(0,omega)*mu)

        elif self.delay_dist == 'truncated_gaussian':
            a0 = self.Phi(-mu/sigma+1j*omega*sigma)
            a1 = self.Phi(-mu/sigma)
            b0 = np.exp(-0.5*np.power(sigma*omega,2))
            b1 = np.exp(-complex(0,omega)*mu)
            return (1.0-a0)/(1.0-a1)*b0*b1

        elif self.delay_dist == 'gaussian':
            b0 = np.exp(-0.5*np.power(sigma*omega,2))
            b1 = np.exp(-complex(0,omega)*mu)
            return b0*b1

    def create_H(self, omega):
        ''' Returns matrix incorporating the transfer function and
        the instantaneous rate jumps at frequency omega.
        '''
        # factor due to weight scaling of NEST in current equation
        # of 'iaf_psc_exp'-model
        fac = 2*self.tauf/self.C
        taum = 1e-3*self.taum
        tauf = 1e-3*self.tauf
        Delay_dist = self.create_delay_dist_matrix(omega)
        if self.tf_mode == 'analytical':
            # find nearest omega, important when the transfer function is
            # read from file
            k = np.argmin(abs(self.omegas-np.abs(omega.real)))
            trans_func = np.transpose(self.trans_func)
            if omega < 0:
                trans_func = np.conjugate(trans_func)
            H = taum*fac*trans_func[k]/complex(1,omega*tauf)
            H = np.hstack([H for i in range(self.dimension)])
            H = np.transpose(H.reshape(self.dimension,self.dimension))
        else:
            H = np.zeros((self.dimension,self.dimension),dtype=complex)
            tau_I = self.tau_impulse_p*0.001
            tau_E = self.tau_impulse_n*0.001
            H[0:self.dimension+1:2] = 1/complex(1.0,omega*tau_I)
            H[1:self.dimension+1:2] = 1/complex(1.0,omega*tau_E)
            H = self.H_df*np.transpose(H)
        H *= Delay_dist
        return H

    def create_MH(self, omega):
        ''' Returns effective connectivity matrix.'''
        H = self.create_H(omega)
        MH = H*self.M
        return MH

    def spec(self, omega):
        '''Returns vector of power spectra for all populations at
        frequency omega.
        '''
        MH_plus = self.create_MH(omega)
        Q_plus = np.linalg.inv(np.identity(self.dimension)-MH_plus)
        C = np.dot(Q_plus,np.dot(self.D,np.transpose(np.conjugate(Q_plus))))
        return np.power(abs(np.diag(C)),2)

    def spec_approx(self, omega):
        '''Returns vector of power spectra approximation by the
        dominant eigenmode for all populations at frequency omega.
        '''
        MH_plus = self.create_MH(omega)
        ep, Up = np.linalg.eig(MH_plus)
        Up_inv = np.linalg.inv(Up)
        # find eigenvalue closest to one
        index = np.argmin(np.abs(ep-1))
        # approximate effective connectiviy by outer product of
        # eigenvectors (corresponding to the dominant mode) weightes
        # by the eigenvalue
        MH_plus = ep[index]*np.outer(Up[:,index],Up_inv[index])
        Q_plus = np.linalg.inv(np.identity(self.dimension)-MH_plus)
        C = np.dot(Q_plus,np.dot(self.D,np.transpose(np.conjugate(Q_plus))))
        return np.power(abs(np.diag(C)),2)

    def power_spectrum(self):
        '''Returns power spectra for all populations.
        Shape of output: (len(self.populations), len(self.omegas))
        '''
        power = np.asarray(map(self.spec, self.omegas))
        return self.omegas/(2.0*np.pi), np.transpose(power)

    def power_spectrum_approx(self):
        '''Returns power spectra approximation by the dominant
        eigenmode for all populations.
        Shape of output: (len(self.populations), len(self.omegas))
        '''
        power = np.asarray(map(self.spec_approx, self.omegas))
        return self.omegas/(2.0*np.pi), np.transpose(power)

    def get_eigenvalues(self, matrix, omega):
        """Returns vector of eigenvalues of matrix at frequency omega.

        Arguments:
        matrix: string, options are 'MH', 'prop' and 'prop_inv'
        omega: frequency as 2*pi*f, with f in Hz
        """
        MH = self.create_MH(omega)
        if matrix == 'MH':
            eig, vr = np.linalg.eig(MH)
            return eig

        Q = np.linalg.inv(np.identity(self.dimension) - MH(omega))
        P = np.dot(Q(omega), MH(omega))
        if matrix == 'prop':
            eig, vr = np.linalg.eig(P)
        elif matrix == 'prop_inv':
            eig, vr = np.linalg.eig(np.linalg.inv(P))

        return eig

    def eigenvalue_spectrum(self, matrix):
        '''Returns eigenvalue spectra for matrix.

        Arguments:
        matrix: string, options are 'MH', 'prop' and 'prop_inv'

        Returns:
        vector of frequencies in Hz
        eigenvalues of shape (len(self.populations), len(self.omegas))
        '''
        eigs = [self.get_eigenvalues(matrix, w) for w in self.omegas]
        eigs = np.transpose(np.asarray(eigs))
        return self.omegas/(2.0*np.pi), eigs
