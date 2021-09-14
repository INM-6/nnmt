from .network import Network


class Plain(Network):

    def __init__(self, network_params=None, analysis_params=None, file=None):

        super().__init__(network_params, analysis_params, file)

        self._convert_param_dicts_to_base_units_and_strip_units()

    def _instantiate(self, new_network_params, new_analysis_params):
        return Plain._instantiate(new_network_params, new_analysis_params)