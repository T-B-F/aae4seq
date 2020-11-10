#!/usr/bin/env python

import os, sys, argparse
import hdbscan
import pandas as pds
import numpy as np
import matplotlib.pyplot as plt

def get_cmd():
    parser = argparse.ArgumentParser()
    
    fileargs = parser.add_argument_group('Input files')
    fileargs.add_argument("-i", action="store", dest="inputfile")
    fileargs.add_argument("-o", action="store", dest="outputfile")

    clsargs = parser.add_argument_group('Bayesian Gaussian Mixture')
    clsargs.add_argument("--min_cluster_size", action="store", dest="min_cluster_size", default=30, type=int)
    clsargs.add_argument("--min_samples", action="store", dest="min_samples", type=int, default=-1)
    clsargs.add_argument("--metric", action="store", dest="metric", default="euclidean")

    params = parser.parse_args()
    if params.min_samples < 0:
        params.min_samples = None
    return params


def main():
    params = get_cmd()

    X = np.load(params.inputfile)["arr_0"]

    clusterer = hdbscan.HDBSCAN(metric=params.metric, 
                                min_cluster_size=params.min_cluster_size,
                                min_samples=params.min_samples)
    clusterer.fit(X)
    print("Number of unique label: {}".format(len(np.unique(clusterer.labels_))))
    
    with open(params.outputfile, "w") as outf:
        for i in range(len(clusterer.labels_)):
            label = clusterer.labels_[i]
            outf.write("{} {}\n".format(i, label))


    sys.exit(0)

if __name__ == "__main__":
    main()

