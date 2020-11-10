#!/usr/bin/env python
""" visualize latent space
"""

import os, sys, argparse
import numpy as np
import scipy as scp
import sklearn
import matplotlib.pyplot as plt
import ete3
from scipy.cluster import hierarchy

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", dest="inputfile")
    parser.add_argument("-l", action="store", dest="labelfile")
    parser.add_argument("-o", action="store", dest="outputfile")
    parser.add_argument("-p", action="store", dest="plotfile")
    params = parser.parse_args()
    return params

def linkage_to_newick(Z, labels):
    tree = hierarchy.to_tree(Z)
    def buildNewick(node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return "{}:{}{}".format(leaf_names[node.id],(parentdist - node.dist)/2, newick)
        else:
            if len(newick) > 0:
                newick = "):{}{}".format((parentdist - node.dist)/2, newick)
            else:
                newick = ");"
            newick = buildNewick(node.get_left(), newick, node.dist, leaf_names)
            newick = buildNewick(node.get_right(), ",{}".format(newick), node.dist, leaf_names)
            newick = "({}".format(newick)
        return newick
    return buildNewick(tree, "", tree.dist, labels)

def main():
    params = get_cmd()

    X = np.load(params.inputfile)
    if params.labelfile is not None:
        labels = list()
        with open(params.labelfile) as inf:
            for line in inf:
                tmp = line.split()
                labels.append("{}_{}".format(tmp[0], tmp[1]))
    else:
        labels = np.arange(len(X))

    Z = hierarchy.linkage(X, 'ward')

    nwk_tree = linkage_to_newick(Z, labels)
    tree_ete3 = ete3.Tree(nwk_tree)

    tree_ete3.write(outfile=params.outputfile)


    if params.plotfile:
        hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=labels)
        plt.savefig(params.plotfile, dpi=300)
    

    sys.exit(0)

if __name__ == "__main__":
    main()

