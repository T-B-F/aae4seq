from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os, sys

from keras import backend as K

import pandas as pd
import numpy as np 
import io

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
    io.write_num_sequences(filename, sequences, idx2aa, step)

def concatenate_input(noise_input, noise_label=None, noise_codes=None):
    if noise_label is not None:
        noise_input = [noise_input, noise_label]
        if noise_codes is not None:
            noise_input += noise_codes        
    return noise_input
