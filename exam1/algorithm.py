import random
import numpy as np

class Detector:
    def __init__(self):
        self.data_all = []

    def fit_predict(self, ptr):
        self.data_all.append(ptr)

        q1 = np.quantile(self.data_all[-100:], 0.25)
        q3 = np.quantile(self.data_all[-100:], 0.7)
        iqr = q3 - q1
        
        lowBound = q1 - 1.5 * iqr
        upBound = q3 + 1.5 * iqr
        pred = random.choices([0, 1], weights=[0.99, 0.01])[0]
        return 0 if upBound >= ptr >= lowBound else 1
