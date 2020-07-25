import os
import csv
import shutil
import zipfile
import pandas as pd

from algorithm import Detector
from plot import plot


def detect(path, detector):
    pred = []
    vals = []
    data_pd = pd.read_csv(path, index_col=0)
    q1 = data_pd['value'].quantile(.25)
    q3 = data_pd['value'].quantile(.75)
    iqr = q3 -q1
    lowBound = q1 - 1.5 * iqr
    upBound = q3 + 1.5 * iqr

    with open(path, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            val = row.get('value')
            ret = detector.fit_predict(float(val), lowBound, upBound)
            pred.append(float(ret))
            vals.append(float(val))
    return vals, pred


if __name__ == '__main__':

    if not os.path.exists('result'):
        os.mkdir('result')

    zip_path = 'data/time-series.zip'
    archive = zipfile.ZipFile(zip_path, 'r')

    for i in range(1, 68):
        detector = Detector()

        name = f'time-series/real_{i}.csv'
        path = archive.extract(member=name)

        vals, pred = detect(path, detector)
        plot(name, vals, pred)

    shutil.rmtree('time-series')
