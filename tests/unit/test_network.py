import pytest


class Test_initialization:
    
    @pytest.mark.parametrize('params', ['netw', 'analys'])
    def test_parameters_updated_on_initialization(self, params):
        pass
    
    def test_warning_is_given_if_necessary_parameters_are_missing(self):
        pass
    
    def test_if_derive_params_false_no_calculation_of_derived_params(self):
        pass
    
    def test_calculation_of_dependent_network_parameters(self):
        """Split?"""
        pass
    
    def test_calculate_dependent_analysis_parameters(self):
        """Split?"""
        pass
    
    def test_hash(self):
        pass
    
    def test_loading_of_existing_results(self):
        pass
    
    def test_change_of_parameters(self):
        pass
    
    
class Test_check_and_store_decorator:
    
    def test_check_and_store(self):
        """Very complicated!"""
        pass
    
    
class Test_saving_and_loading_routines:
    
    def test_saving(self):
        pass
    
    def test_loading(self):
        pass
    

class Test_functionality:
    
    @pytest.mark.parametrize('func', ['all_functions'])
    def test_correct_functions_are_called(self):
        pass
