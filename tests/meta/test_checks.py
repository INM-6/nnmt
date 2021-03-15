import pytest

from ..checks import pint_wrap

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
