from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re, sys
import pandas as pds
import numpy as np
from scipy.stats import spearmanr
import tensorflow as tf

from BioUtils import BUio


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def read_data(inputfile, remove_gap=False, get_labels=False):
    sequences = list()
    labels = list()
    max_size = 0
    for record in BUio.read_fastabioit(inputfile):
        name = record.id
        if get_labels:
            ids = int(name.split("_")[1])
            if ids == 13:
                ids = 12
        seq = str(record.seq).upper()
        if remove_gap:
            seq = seq.replace("-", "")
        if len(seq) > max_size:
            max_size = len(seq)
        sequences.append(seq)
        if get_labels:
            labels.append(ids)
    labels = np.asarray(labels)
    return sequences, labels, max_size
            
            
def read_weights(inputfile):
    data = np.load(inputfile)
    return data

def write_num_sequences(filename, sequences, idx2aa, step):
    with open(filename, "w") as outf:
        for i in range(len(sequences)):
            outf.write(">seq_{:06d}_{}\n".format(step ,i))
            gen_seq = sequences[i]
            num_seq = gen_seq.argmax(axis=1)
            new_seq = "".join(idx2aa[idx] for idx in num_seq)
            new_seq = new_seq.replace("^", "-").replace("$", "-")
            outf.write("{}\n".format(new_seq))
    
def write_info(filename, noise_input, predictions):
    np.savez(filename, z_input=noise_input[0], labels=noise_input[1], 
             code1=noise_input[2], code2=noise_input[3],
             real_fake_preds = predictions[0].flatten(), 
             label_preds = predictions[1])
             #code1_preds = predictions[2].flatten(), 
             #code2_preds = predictions[3].flatten())
