from numpy.testing import assert_array_equal


def assert_quantity_array_equal(qarray1, qarray2):
    assert_array_equal(qarray1, qarray2)
    assert qarray1.units == qarray2.units
    
    
def assert_dict_of_quantity_arrays_equal(dict1, dict2):
    for key in set(dict1) | set(dict2):
        assert_quantity_array_equal(dict1[key], dict2[key])
        
        
def save_and_load(network, tmpdir):
    temp = tmpdir.mkdir('temp')
    with temp.as_cwd():
        network.save(file_name='test.h5')
        network.load(file_name='test.h5')


class Test_Network_functions_give_correct_results:
    
    def test_firing_rates(self, network, std_results):
        firing_rates = network.firing_rates()
        assert_quantity_array_equal(firing_rates, std_results['firing_rates'])
        
    def test_mean_input(self, network, std_results):
        mean_input = network.mean_input()
        assert_quantity_array_equal(mean_input, std_results['mean_input'])
        
    def test_standard_deviation(self, network, std_results):
        std_input = network.std_input()
        assert_quantity_array_equal(std_input, std_results['std_input'])
    
    def test_working_point(self, network, std_results):
        working_point = network.working_point()
        expected_working_point = dict(firing_rates=std_results['firing_rates'],
                                      mean_input=std_results['mean_input'],
                                      std_input=std_results['std_input'])
        assert_dict_of_quantity_arrays_equal(working_point,
                                             expected_working_point)
    
    def test_delay_dist_matrix(self, network, std_results):
        ddm = network.delay_dist_matrix()
        assert_quantity_array_equal(ddm, std_results['delay_dist'])
    
    def test_delay_dist_matrix_single(self, network, std_results):
        ddm = network.delay_dist_matrix(network.analysis_params['omega'])
        assert_quantity_array_equal(ddm, std_results['delay_dist_single'][0])
    
    def test_transfer_function(self, network, std_results):
        transfer_fn = network.transfer_function()
        assert_quantity_array_equal(transfer_fn,
                                    std_results['transfer_function'])
    
    def test_transfer_function_single(self, network, std_results):
        transfer_fn = network.transfer_function(
            network.analysis_params['omega'])
        assert_quantity_array_equal(transfer_fn,
                                    std_results['transfer_function_single'][0])
    
    def test_sensitivity_measure(self, network, std_results):
        sm = network.sensitivity_measure(network.analysis_params['omega'])
        assert_quantity_array_equal(sm, std_results['sensitivity_measure'])
    
    def test_power_spectra(self, network, std_results):
        ps = network.power_spectra()
        assert_quantity_array_equal(ps, std_results['power_spectra'])
    
    def test_eigenvalue_spectra(self, network, std_results):
        es = network.eigenvalue_spectra('MH')
        assert_quantity_array_equal(es, std_results['eigenvalue_spectra'])
    
    def test_r_eigenvec_spectra(self, network, std_results):
        es = network.r_eigenvec_spectra('MH')
        assert_quantity_array_equal(es, std_results['r_eigenvec_spectra'])
    
    def test_l_eigenvec_spectra(self, network, std_results):
        es = network.l_eigenvec_spectra('MH')
        assert_quantity_array_equal(es, std_results['l_eigenvec_spectra'])
    
    def test_additional_rates_for_fixed_input(self, network, std_results):
        mean_input_set = network.network_params['mean_input_set']
        std_input_set = network.network_params['std_input_set']
        nu_e_ext, nu_i_ext = network.additional_rates_for_fixed_input(
            mean_input_set, std_input_set)
        assert_quantity_array_equal(nu_e_ext, std_results['nu_e_ext'])
        assert_quantity_array_equal(nu_i_ext, std_results['nu_i_ext'])


class Test_saving_and_loading:
    
    def test_firing_rates(self, network, tmpdir):
        saved = network.firing_rates()
        save_and_load(network, tmpdir)
        loaded = network.results['firing_rates']
        assert_quantity_array_equal(saved, loaded)
        
    def test_mean_input(self, network, tmpdir):
        saved = network.mean_input()
        save_and_load(network, tmpdir)
        loaded = network.results['mean_input']
        assert_quantity_array_equal(saved, loaded)
        
    def test_standard_deviation(self, network, tmpdir):
        saved = network.std_input()
        save_and_load(network, tmpdir)
        loaded = network.results['std_input']
        assert_quantity_array_equal(saved, loaded)
    
    def test_working_point(self, network, tmpdir):
        saved = network.working_point()
        save_and_load(network, tmpdir)
        loaded_fr = network.results['firing_rates']
        loaded_mean = network.results['mean_input']
        loaded_std = network.results['std_input']
        loaded = dict(firing_rates=loaded_fr,
                      mean_input=loaded_mean,
                      std_input=loaded_std)
        assert_dict_of_quantity_arrays_equal(saved, loaded)
    
    def test_delay_dist_matrix(self, network, tmpdir):
        saved = network.delay_dist_matrix()
        save_and_load(network, tmpdir)
        loaded = network.results['delay_dist']
        assert_quantity_array_equal(saved, loaded)
    
    def test_delay_dist_matrix_single(self, network, tmpdir):
        saved = network.delay_dist_matrix(network.analysis_params['omega'])
        save_and_load(network, tmpdir)
        loaded = network.results['delay_dist_single'][0]
        assert_quantity_array_equal(saved, loaded)
    
    def test_transfer_function(self, network, tmpdir):
        saved = network.transfer_function()
        save_and_load(network, tmpdir)
        loaded = network.results['transfer_function']
        assert_quantity_array_equal(saved, loaded)
    
    def test_transfer_function_single(self, network, tmpdir):
        saved = network.transfer_function(network.analysis_params['omega'])
        save_and_load(network, tmpdir)
        loaded = network.results['transfer_function_single'][0]
        assert_quantity_array_equal(saved, loaded)
    
    def test_sensitivity_measure(self, network, tmpdir):
        sm_saved = network.sensitivity_measure(
            network.analysis_params['omega'])
        save_and_load(network, tmpdir)
        sm_loaded = network.results['sensitivity_measure']
        assert_quantity_array_equal(sm_saved, sm_loaded)
    
    def test_power_spectra(self, network, tmpdir):
        saved = network.power_spectra()
        save_and_load(network, tmpdir)
        loaded = network.results['power_spectra']
        assert_quantity_array_equal(saved, loaded)
    
    def test_eigenvalue_spectra(self, network, tmpdir):
        saved = network.eigenvalue_spectra('MH')
        save_and_load(network, tmpdir)
        loaded = network.results['eigenvalue_spectra']
        assert_quantity_array_equal(saved, loaded)
    
    def test_r_eigenvec_spectra(self, network, tmpdir):
        saved = network.r_eigenvec_spectra('MH')
        save_and_load(network, tmpdir)
        loaded = network.results['r_eigenvec_spectra']
        assert_quantity_array_equal(saved, loaded)
    
    def test_l_eigenvec_spectra(self, network, tmpdir):
        saved = network.l_eigenvec_spectra('MH')
        save_and_load(network, tmpdir)
        loaded = network.results['l_eigenvec_spectra']
        assert_quantity_array_equal(saved, loaded)
    
    def test_additional_rates_for_fixed_input(self, network, tmpdir):
        mean_input_set = network.network_params['mean_input_set']
        std_input_set = network.network_params['std_input_set']
        saved_e, saved_i = network.additional_rates_for_fixed_input(
            mean_input_set, std_input_set)
        save_and_load(network, tmpdir)
        loaded_e = network.results['nu_e_ext']
        loaded_i = network.results['nu_i_ext']
        assert_quantity_array_equal(saved_e, loaded_e)
        assert_quantity_array_equal(saved_i, loaded_i)
