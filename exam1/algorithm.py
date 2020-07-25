import random


class Detector:
    def fit_predict(self, ptr, lowBound, upBound):
        # pred = random.choices([0, 1], weights=[0.99, 0.01])[0]
        return 0 if upBound >= ptr >= lowBound else 1