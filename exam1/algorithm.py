import random
import numpy as np

class Detector:
    def __init__(self):
        self.data_all = []
        self.prd = []

    def fit_predict(self, ptr):
        self.data_all.append(ptr)

        q1 = np.quantile(self.data_all[-50:], 0.25)
        q3 = np.quantile(self.data_all[-50:], 0.75)
        iqr = q3 - q1
        
        lowBound = q1 - 1.5 * iqr
        upBound = q3 + 1.5 * iqr

        if len(self.prd) and self.prd[-1] == 1:
            self.prd.append(0)
            return 0
        elif upBound >= ptr >= lowBound:
            self.prd.append(0)
            return 0
        else :
            self.prd.append(1)
            return 1
        # return 0 if upBound >= ptr >= lowBound else 1
