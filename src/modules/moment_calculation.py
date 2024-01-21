
import numpy as np

class MomentCalculation:
    def __init__(self, npy_file):
        self.my_array = np.load(npy_file)

    def my_mean(self):
        return np.mean(self.my_array)
    
    def my_stdeav(self):
        return np.std(self.my_array)