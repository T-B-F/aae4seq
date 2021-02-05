from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os, sys

import pandas as pd
import numpy as np 
import gzip
import src.io as sio

#Invariants
ORDER_KEY="XILVAGMFYWEDQNHCRKSTPBZ-"[::-1]
ORDER_LIST=list(ORDER_KEY)

def translate_string_to_one_hot(sequence,order_list):
    out=np.zeros((len(order_list),len(sequence)))
    for i in range(len(sequence)):
        out[order_list.index(sequence[i])][i]=1
    return out


#compute distance between two aligned sequences
def aligned_dist(s1,s2):
    count=0
    for i,j in zip(s1,s2):
        if i!=j:
            count+=1
    return count


#Compute a new weight for sequence based on similarity threshold theta 
def reweight_sequences(dataset,theta):
    weights=[1.0 for i in range(len(dataset))]
    start = time.process_time()

    for i in range(len(dataset)):

        if i%250==0:
            print(str(i)+" took "+str(time.process_time()-start) +" s ")
            start = time.process_time()

        for j in range(i+1,len(dataset)):
            if aligned_dist(dataset[i],dataset[j])*1./len(dataset[i]) <theta:
               weights[i]+=1
               weights[j]+=1
    return list(map(lambda x:1./x, weights))
    
    

def translate_string(seq, alphabet):
    out = list()
    for i in range(len(seq)):
        out.append(alphabet.index(seq[i]))
    return out

def data_to_onehot(data):
    #Encode training data in one_hot vectors
    training_data=[]
    labels=[]
    for i, row in data.iterrows():
        translated_seq = translate_string(row["seq"], ORDER_LIST)
        training_data.append(translated_seq)

    training_data_padded = pad_sequences(training_data, padding="post", value=0, dtype="int32")
    training_data_one_hot = to_categorical(training_data_padded)
    return training_data_one_hot

def to_numeric(seq, aa2idx):
    numeric = list()
    for i, s in enumerate(seq):
        try:
            num = [aa2idx[aa] for aa in s]
        except:
            print("Error encoding sequence, n° {}:".format(i+1))
            print(s)
            sys.exit(1)
        numeric.append(num)
    return numeric

def rnd_sequence(generator,
                 idx2aa,
                 noise_input, 
                 noise_label=None,
                 noise_codes=None,
                 step=0,
                 model_name="gan"):
    """Generate fake sequences
    """
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.fasta" % step)
    sequences = generator.predict(noise_input)
    sio.write_num_sequences(filename, sequences, idx2aa, step)

def concatenate_input(noise_input, noise_label=None, noise_codes=None):
    if noise_label is not None:
        noise_input = [noise_input, noise_label]
        if noise_codes is not None:
            noise_input += noise_codes        
    return noise_input

class SeqRecord():
    """ protein sequence handler """
    def __init__(self):
        self.id= ""
        self.seq = ""


def read_fastabioit(path):
    """ read fasta file, one sequence after an other

    Arguments:
    ==========
    path: string
        path of fasta file

    Return:
    =======
    record: SeqRecord
        a record of a protein sequence
    """
    ext = os.path.splitext(path)[1]
    record = SeqRecord()
    if ext in [".gzip", ".gz"]:
        with gzip.open(path, 'rt', encoding="utf-8") as handle:
            for line in handle:
                if line[0] == ">":
                    if record.id != "":
                        yield record
                    record = SeqRecord()
                    record.id=line[1:].split()[0]
                else:
                    record.seq += line.strip()
        yield record
    else:
        with open(path) as handle:
            for line in handle:
                if line[0] == ">":
                    if record.id != "":
                        yield record
                    record = SeqRecord()
                    record.id=line[1:].split()[0]
                else:
                    record.seq += line.strip()
        yield record

def read_data(inputfile, remove_gap=False, get_labels=False):
    """ read fasta file

    Arguments:
    ==========
    inputfile: string
        path of the fasta file
    remove_gap: bool
        if True remove gap else keep them (default False)
    get_labels: bool
        return labels if True (default False)

    Return:
    =======
    sequences: list
        list of protein sequences as string
    labels: array
        array of label id as written in the fasta record id 
        if get_labels is True, else empty array
    max_size: int
        longest protein sequence
    """
    sequences = list()
    labels = list()
    max_size = 0
    for record in read_fastabioit(inputfile):
        name = record.id
        if get_labels:
            ids = int(name.split("_")[1])
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
