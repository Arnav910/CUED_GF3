import numpy as np

class FIR:
    def __init__(self, filter_coeff: list[float]|np.ndarray):
        if not isinstance(filter_coeff, np.ndarray):
            filter_coeff = np.array(filter_coeff)
        self.coefficients = filter_coeff

    def get_frequency(self,bins):
        pass
    