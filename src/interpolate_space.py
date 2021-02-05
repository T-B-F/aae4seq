#!/usr/bin/env python
""" perform spherical interpolation between points 
and run the decoder of the VAE  on the interpolated data
"""
import os, sys, argparse
import numpy as np
import keras
from BioUtils import BUio
from src.utils import read_data

ORDER_KEY="XILVAGMFYWEDQNHCRKSTPBZ-"[::-1]
ORDER_LIST=list(ORDER_KEY)

def warm_prediction(a, temperature=1.0):
    """ warmup prediction based on temperature factor

    Arguments
    =========
    a: array
        prediction values
    temperature: float
        scaling factor

    Return
    ======
    sample_temp: array
        new prediction values
    """

    a = (np.array(a)**(1/temperature)).astype(float)
    p_sum = a.sum(axis=0)
    p_sum[p_sum == 0] = 1e-6
    sample_temp = a/p_sum 
    return sample_temp

def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1.
    Arguments:
    ==========
    val: float
        interpolated step between 0 and 1
    low: float
        starting value
    high: float
        end value

    Return
    ======
    value: float
        interpolated value
    """
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", dest="inputfile", help="fasta file")
    parser.add_argument("-c", action="store", dest="coords", help="coordinates of sequences encoded by model")
    parser.add_argument("-d", action="store", dest="decoder", help="decoder weight")
    parser.add_argument("-q", action="store", dest="query", type=int, help="query index, ie seq number in fasta and coordinate files")
    parser.add_argument("-t", action="store", dest="target", type=int, help="target index, ie seq number in fasta and coordiante files")
    parser.add_argument("-o", action="store", dest="outputfile", help="output file in fasta format")
    parser.add_argument("-s", action="store", dest="steps", type=int, default=50, help="number of interpolation steps")
    params = parser.parse_args()
    return params

def main():
    params = get_cmd()

    aa2idx, idx2aa = dict(), dict()
    for i, aa in enumerate(ORDER_LIST):
        aa2idx[aa] = i
        idx2aa[i] = aa

    X = np.load(params.coords)["arr_0"]
    sequences, labels, _ = read_data(params.inputfile, get_labels=True)
    
    assert params.query >= 0 and params.query < len(sequences)
    assert params.target >= 0 and params.target < len(sequences)
    assert params.query != params.target
    if labels[params.query] == labels[params.target]:
        print("Warning, query and target are from the same family")

    seq_query = sequences[params.query]
    seq_target = sequences[params.target]
    
    encoded_query = X[params.query]
    encoded_target = X[params.target]

    decoder = keras.models.load_model(params.decoder)

    points = list()
    for v in np.linspace(0, 1, params.steps+2):
        points.append(slerp(v, encoded_query, encoded_target))

    points = np.asarray(points)
    decoded_points = decoder.predict(points)
    decoded_seq = []
    for pred in decoded_points:
        wp = warm_prediction(pred.T, 0.5).T
        num_seq = [np.random.choice(np.arange(len(ORDER_LIST)), p=wp[j]) for j in range(len(wp))] 
        decoded_seq.append("".join(idx2aa[i] for i in num_seq))

    with open(params.outputfile, "w") as outf:
        #outf.write(">query_original\n{}\n".format(seq_query))
        outf.write(">query\n{}\n".format(decoded_seq[0]))
        for i in range(1, len(decoded_seq)-1):
            outf.write(">interpolated_{}\n{}\n".format(i, decoded_seq[i]))
        outf.write(">target\n{}\n".format(decoded_seq[-1]))
        #outf.write(">target_original\n{}\n".format(seq_target))

    sys.exit(0)

if __name__ == "__main__":
    main()

