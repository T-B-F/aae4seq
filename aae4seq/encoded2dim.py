#!/usr/bin/env python

import os, sys, argparse
import io, utils
import numpy as np
import keras

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", dest="inputfile")
    parser.add_argument("-e", action="store", dest="encoderfile")
    parser.add_argument("-o", action="store", dest="outputfile")
    params = parser.parse_args()
    return params

def main():

    params = get_cmd()

    aa2idx = dict()
    idx2aa = dict()
    for i, aa in enumerate(utils.ORDER_LIST):
        aa2idx[aa] = i
        idx2aa[i] = aa

    print("read sequences")
    sequences, labels, _ = io.read_data(params.inputfile)
    print("to numeric")
    x_train = utils.to_numeric(sequences, aa2idx)
    print("to categorical")
    x_train = keras.utils.to_categorical(x_train, num_classes=len(utils.ORDER_LIST))

    print("load encoder")
    encoder = keras.models.load_model(params.encoderfile)
    print("encoding ...")
    encoded = encoder.predict(x_train)

    print("save")
    np.savez(params.outputfile, encoded)

    print("done")
    sys.exit(0)

if __name__ == "__main__":
    main()
