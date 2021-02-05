#!/usr/bin/env python

import os, sys, argparse
import matplotlib.pyplot as plt
import numpy as np


def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", dest="inputfile")
    parser.add_argument("-l", action="store", dest="labelfile")
    parser.add_argument("-o", action="store", dest="outputfile")
    params = parser.parse_args()
    return params

def main():

    params = get_cmd()

    X = np.load(params.inputfile)["arr_0"]

    fig, ax = plt.subplots()
    if params.labelfile:
        labels = list()
        with open(params.labelfile) as inf:
            for line in inf:
                tmp = line.split()
                labels.append(int(tmp[1]))
        labels = np.asarray(labels)
        for i in np.unique(labels):
            if not np.any(labels == i):
                continue
            ax.scatter(X[labels == i, 0], X[labels == i, 1], s=4, alpha=0.5, label="cl_{}".format(i))
    else:
        ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.5)
    if params.labelfile and len(np.unique(labels)) < 15:

        plt.legend(bbox_to_anchor=[1, 1, 0, 0])
        plt.tight_layout()
    plt.savefig(params.outputfile, dpi=300)
    plt.show()
        

    sys.exit(0)

if __name__ == "__main__":
    main()
