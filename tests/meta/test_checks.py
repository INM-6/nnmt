import pytest

from ..checks import (pint_wrap,
                      check_dict_contains_no_quantity,
                      check_dict_contains_no_val_unit_dict)

import lif_meanfield_tools as lmt
ureg = lmt.ureg


class Test_pint_wrapper:
    
    def test_only_units_of_first_two_arguments_are_stripped_off(self):
        def test_function(a, b, c):
            return a, b, c
        test_function = pint_wrap(test_function)
        a0 = 10 * ureg.mV
        b0 = 20 * ureg.mV
        c0 = 30 * ureg.mV
        a1, b1, c1 = test_function(a0, b0, c0)
        with pytest.raises(AttributeError):
            a1.units
        with pytest.raises(AttributeError):
            b1.units
        c1.units
        
    def test_arguments_are_passed_correctly_to_wrapped_function(self):
        def test_function(a, b, c):
            return a, b, c
        test_function = pint_wrap(test_function)
        a0 = 10 * ureg.mV
        b0 = 20 * ureg.mV
        c0 = 30 * ureg.mV
        a1, b1, c1 = test_function(a0, b0, c0)
        assert a0.magnitude == a1
        assert b0.magnitude == b1
        assert c0.magnitude == c1.magnitude
        assert c0.units == c1.units
        
    
class Test_check_dict_contains_no_quantity:
    
    def test_simple_dict_with_no_quantities(self):
        test = dict(a=1, b=[1, 2, 3], c='ham')
        check_dict_contains_no_quantity(test)
    
    def test_simple_dict_with_quantities(self):
        test = dict(a=1 * ureg.Hz, b=[1, 2, 3], c='ham')
        with pytest.raises(AssertionError):
            check_dict_contains_no_quantity(test)
        
    def test_dict_of_dict_with_no_quantities(self):
        test = dict(a=dict(a1=1, a2='ham', a3=[1, 2]),
                    b=dict(b1=2, b2='spam', b3=[3, 4]))
        check_dict_contains_no_quantity(test)
        
    def test_dict_of_dict_with_quantities(self):
        test = dict(a=dict(a1=1 * ureg.Hz, a2='ham', a3=[1, 2]),
                    b=dict(b1=2, b2='spam', b3=[3, 4]))
        with pytest.raises(AssertionError):
            check_dict_contains_no_quantity(test)
        
    def test_dict_of_dict_of_dict_with_quantities(self):
        test = dict(a=dict(a1=1 * ureg.Hz,
                           a2=dict(a3=[1, 2, 3] * ureg.ms),
                    b=dict(b1=2, b2='spam', b3=[3, 4])))
        with pytest.raises(AssertionError):
            check_dict_contains_no_quantity(test)
    
    
class Test_check_dict_contains_no_val_unit_dict:
    
    def test_simple_dict_with_no_val_unit_dict(self):
        test = dict(a=1, b=[1, 2, 3], c='ham')
        check_dict_contains_no_val_unit_dict(test)
    
    def test_simple_val_unit_dict(self):
        test = dict(val=1, unit='hertz')
        with pytest.raises(AssertionError):
            check_dict_contains_no_val_unit_dict(test)
        
    def test_dict_of_dict_with_no_val_unit_dict(self):
        test = dict(a=dict(a1=1, a2='ham', a3=[1, 2]),
                    b=dict(b1=2, b2='spam', b3=[3, 4]))
        check_dict_contains_no_val_unit_dict(test)
        
    def test_dict_of_dict_with_val_unit_dict(self):
        test = dict(a=dict(a1=dict(val=1, unit='hertz'), a2='ham', a3=[1, 2]),
                    b=dict(b1=2, b2='spam', b3=[3, 4]))
        with pytest.raises(AssertionError):
            check_dict_contains_no_val_unit_dict(test)
        
    def test_dict_of_dict_of_dict_with_val_unit_dict(self):
        test = dict(a=dict(a1=1 * ureg.Hz,
                           a2=dict(a3=dict(val=1, unit='hertz')),
                    b=dict(b1=2, b2='spam', b3=[3, 4])))
        with pytest.raises(AssertionError):
            check_dict_contains_no_val_unit_dict(test)
