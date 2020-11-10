#!/usr/bin/env python
""" gaussian mixture model over samples
"""

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import os, sys, argparse
import pandas as pds
import numpy as np
import matplotlib.pyplot as plt

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", dest="inputfile")
    parser.add_argument("-o", action="store", dest="outputfile")
    params = parser.parse_args()
    return params


def main():
    params = get_cmd()

    X = np.load(params.inputfile)["arr_0"]

    gmm = BayesianGaussianMixture(max_iter=500, n_components=40, covariance_type="full")
    gmm.fit(X)
    best_gmm = gmm
    Y = best_gmm.predict(X)
    
    with open(params.outputfile, "w") as outf:
        for i in range(len(Y)):
            outf.write("{} {}\n".format(i, Y[i]))


    sys.exit(0)

if __name__ == "__main__":
    main()

