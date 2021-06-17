import pytest
import numpy as np

from ..checks import (pint_wrap,
                      assert_quantity_array_equal,
                      assert_units_equal,
                      assert_dimensionality_equal,
                      check_quantity_dicts_are_equal,
                      check_dict_contains_no_quantity,
                      check_dict_contains_no_val_unit_dict)

import nnmt
ureg = nnmt.ureg


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
        
    def test_works_even_if_with_mixed_args_are_kwargs(self):
        def test_function(a, b, c):
            return a, b, c
        test_function = pint_wrap(test_function)
        a0 = 10 * ureg.mV
        b0 = 20 * ureg.mV
        c0 = 30 * ureg.mV
        a1, b1, c1 = test_function(a0, b=b0, c=c0)
        assert a0.magnitude == a1
        assert b0.magnitude == b1
        assert c0.magnitude == c1.magnitude
        assert c0.units == c1.units
        
    def test_works_if_function_has_unpassed_standard_parameters(self):
        def test_function(a, b, c, d=1):
            return a, b, c, d
        test_function = pint_wrap(test_function)
        a0 = 10 * ureg.mV
        b0 = 20 * ureg.mV
        c0 = 30 * ureg.mV
        a1, b1, c1, d1 = test_function(a0, b=b0, c=c0)
        assert a0.magnitude == a1
        assert b0.magnitude == b1
        assert c0.magnitude == c1.magnitude
        assert c0.units == c1.units
        assert d1 == 1
        
        
class Test_assert_quantity_array_equal:
    
    def test_fails_when_amplitudes_differ(self):
        arr1 = np.array([1, 2, 3]) * ureg.mV
        arr2 = np.array([1, 2, 4]) * ureg.mV
        with pytest.raises(AssertionError):
            assert_quantity_array_equal(arr1, arr2)

    def test_fails_when_units_differ(self):
        arr1 = np.array([1, 2, 3]) * ureg.mV
        arr2 = np.array([1, 2, 3]) * ureg.ms
        with pytest.raises(AssertionError):
            assert_quantity_array_equal(arr1, arr2)
    
    def test_passes_when_arrays_equal(self):
        arr1 = np.array([1, 2, 3]) * ureg.mV
        assert_quantity_array_equal(arr1, arr1)
        

class Test_assert_units_equal:
    
    def test_fails_when_units_differ(self):
        var1 = 1 * ureg.ms
        var2 = 1 * ureg.mV
        with pytest.raises(AssertionError):
            assert_units_equal(var1, var2)
    
    def test_passes_when_units_same(self):
        var1 = 1 * ureg.ms
        var2 = [1, 2, 3] * ureg.ms
        assert_units_equal(var1, var2)
    
    
class Test_assert_dimensionality_equal:
    
    def test_fails_when_dimensionalities_differ(self):
        var1 = [1, 2, 3, 4] * ureg.V
        var2 = [1, 2, 3] * ureg.s
        with pytest.raises(AssertionError):
            assert_dimensionality_equal(var1, var2)
            
    def test_passes_when_dimensionalities_same(self):
        var1 = [1, 2, 3, 4] * ureg.V
        var2 = [1, 2, 3] * ureg.mV
        assert_dimensionality_equal(var1, var2)
        
    
class Test_check_quantity_dicts_are_equal:
    
    adicts = [dict(a=1, b=[1, 2, 3], c='ham'),
              dict(a=1 * ureg.Hz, b=[1, 2, 3], c='ham'),
              dict(a=dict(a1=1, a2='ham', a3=[1, 2]),
                   b=dict(b1=2, b2='spam', b3=[3, 4])),
              dict(a=dict(a1=1 * ureg.Hz, a2='ham', a3=[1, 2]),
                   b=dict(b1=2, b2='spam', b3=[3, 4])),
              dict(a=dict(a1=1 * ureg.Hz,
                          a2=dict(a3=[1, 2, 3] * ureg.ms),
                   b=dict(b1=2, b2='spam', b3=[3, 4])))]
    # adicts with one different element
    bdicts = [dict(a=1, b=[3, 2, 3], c='ham'),
              dict(a=2 * ureg.Hz, b=[1, 2, 3], c='ham'),
              dict(a=dict(a1=1, a2='spam', a3=[1, 2]),
                   b=dict(b1=2, b2='spam', b3=[3, 4])),
              dict(a=dict(a1=2 * ureg.Hz, a2='ham', a3=[1, 2]),
                   b=dict(b1=2, b2='spam', b3=[3, 4])),
              dict(a=dict(a1=1 * ureg.Hz,
                          a2=dict(a3=[2, 2, 3] * ureg.ms),
                   b=dict(b1=2, b2='spam', b3=[3, 4])))]
    
    ids = ['dict_without_quantities',
           'dict_with_quantities',
           'nested_dict_without_quantities',
           'nested_dict_with_quantities',
           'double_nested_dict_with_quantitites']
    
    @pytest.mark.parametrize('test_dict', adicts, ids=ids)
    def test_works_for_same_dicts(self, test_dict):
        check_quantity_dicts_are_equal(test_dict, test_dict)
    
    @pytest.mark.parametrize('adict, bdict', zip(adicts, bdicts), ids=ids)
    def test_fails_for_differing_dicts(self, adict, bdict):
        with pytest.raises(AssertionError):
            check_quantity_dicts_are_equal(adict, bdict)
            
    
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
