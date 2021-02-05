#!/usr/bin/env python
# coding: utf-8
""" Use an Adversarial AutoEncoder to learn a latent space of specified 
dimension to represent the input sequence diversity 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Generic imports
import numpy as np
import pandas as pds
import sys, os, json
import argparse

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten, Reshape

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

import src.io as sio, src.utils as sutils


class AAE():
    
    def __init__(self, x_train, params):
        self.datasize = x_train.shape[0]
        self.seq_length = x_train.shape[1]
        self.n_char= x_train.shape[2]
        self.sequence_size = self.seq_length * self.n_char
        
        # network parameters
        self.batch_size = params["batch_size"]
        self.latent_dim  = params["latent_size"]
        self.lr = params["lr"]
        
    def save(self, prefix):
        self.encoder.save(os.path.join(prefix, "encoder_protein_function.h5"))
        self.decoder.save(os.path.join(prefix, "decoder_protein_function.h5"))
        self.discriminator.save(os.path.join(prefix, "discriminator_protein_function.h5"))
        self.adversarial.save(os.path.join(prefix, "adversarial_protein_function.h5"))
        
    def load(self, prefix):
        self.encoder = tf.keras.models.load_model(os.path.join(prefix, "encoder_protein_function.h5"))
        self.decoder = tf.keras.models.load_model(os.path.join(prefix, "decoder_protein_function.h5"))
        self.discriminator = tf.keras.models.load_model(os.path.join(prefix, "discriminator_protein_function.h5"))
        self.adversarial = tf.keras.models.load_model(os.path.join(prefix, "adversarial_protein_function.h5"))
        
    def build(self): 
        
        optimizer = Adam(self.lr, 0.5) 
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
        
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        self.encoder.summary()
        self.decoder.summary()
        
        seq = Input(shape=(self.seq_length, self.n_char))
        encoded_repr = self.encoder(seq)
        reconstructed_seq = self.decoder(encoded_repr)
        
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial = Model(seq, [reconstructed_seq, validity])
        self.adversarial.compile(loss=["categorical_crossentropy", 'binary_crossentropy'], 
                                 metrics=["categorical_accuracy", "binary_accuracy"],
                                 loss_weights=[0.9, 0.1],
                                 optimizer=optimizer)
        
        self.adversarial.summary()
    
    def build_encoder(self):
        # Encoder

        seq = Input(shape=(self.seq_length, self.n_char))

        h = Conv1D(32, 7, strides=2)(seq)
        h = Conv1D(32, 7, strides=2)(h)
        h = Flatten()(h)
        h = Dense(256)(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(256)(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = LeakyReLU(alpha=0.2)(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        latent_repr = Lambda(lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
                             output_shape=lambda p: p[0])([mu, log_var])

        return Model(seq, latent_repr)

    def build_decoder(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.sequence_size, activation='softmax'))
        model.add(Reshape((self.seq_length, self.n_char)))
        model.summary()

        z = Input(shape=(self.latent_dim,))
        seq = model(z)

        return Model(z, seq)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

        
    def train(self, x_train, y_train, outputdir, 
              idx2aa, train_steps,  
              train_weights=None, 
              save_interval=100, to_sample=1000):
        """ train discriminator and adversarial networks
        """
        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        z_samping = np.random.normal(size=(to_sample, self.latent_dim))
        

        basedir = os.path.dirname(outputdir)
        if basedir+"/" == outputdir:
            basedir = os.path.dirname(basedir)
        log_path = os.path.join(outputdir,'logs')
        
        writer = tf.summary.create_file_writer(log_path)
        #callback = TensorBoard(log_path)
        #callback.set_model(self.adversarial)
        
        #fake_weights = np.ones(self.batch_size).tolist()
        
        for epoch in range(train_steps):
            ####
            #  Train Discriminator
            
            # Select a random batch of images
            index = np.random.randint(0, self.datasize, size=self.batch_size)
            sequences = x_train[index]
            if train_weights is not None:
                weights = train_weights[index]

            latent_fake = self.encoder.predict(sequences)
            latent_real = np.random.normal(size=(self.batch_size, self.latent_dim))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            if np.random.rand() > 0.8:
                d_loss_real = self.discriminator.train_on_batch(latent_fake, fake)
                d_loss_fake = self.discriminator.train_on_batch(latent_real, valid)

            ####
            #  Train Generator
            
            # Train the generator
            if train_weights is not None:
                for _ in range(1):
                    g_loss = self.adversarial.train_on_batch(sequences, [sequences, valid], sample_weight=[weights, weights])
            else:
                for _ in range(1):
                    g_loss = self.adversarial.train_on_batch(sequences, [sequences, valid])
                    
            #print(self.adversarial.metrics_names)
            #print(g_loss)

            with writer.as_default():
                tf.summary.scalar('d_loss', d_loss[0], step=epoch+1)
                tf.summary.scalar('g_loss', g_loss[0], step=epoch+1)
                tf.summary.scalar('ae', g_loss[1], step=epoch+1)
                tf.summary.scalar('validity', g_loss[2], step=epoch+1)
                tf.summary.scalar('accuracy', g_loss[3], step=epoch+1)

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f (%.2f, %.2f, %.2f)]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1], g_loss[2], g_loss[3]*100))
            
            if (epoch + 1) % save_interval == 0:
                # plot generator images on a periodic basis
                sutils.rnd_sequence(self.decoder, idx2aa, z_samping,
                                       step = (epoch + 1),
                                       model_name=os.path.join(outputdir, "samples"))
        return True

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", help="input MSA")
    parser.add_argument("-o", "--output", action='store', help="Output directory")
    parser.add_argument("--weights", action="store", dest="weightfile")
    
    parser.add_argument("-l", action="store", dest="latentdim", type=int, default=32)
    parser.add_argument("-n", action="store", dest="niter", type=int, default=500)
    parser.add_argument("-b", action="store", dest="batch_size", type=int, default=84)
    parser.add_argument("-r", action="store", dest="learning_rate", type=float, default=0.001)

    parser.add_argument("--save_interval", action="store", dest="save_interval", type=int, default=100)
    parser.add_argument("--to_sample", action="store", dest="to_sample", type=int, default=1000)
    params = parser.parse_args()
    
    return params

def main(params):
    aa2idx = dict()
    idx2aa = dict()
    for i, aa in enumerate(sutils.ORDER_LIST):
        aa2idx[aa] = i
        idx2aa[i] = aa
    
    weights = None
    if params.weightfile:
        weights = sio.read_weights(params.weightfile)
        
    sequences, labels, _ = sio.read_data(params.inputfile)
    x_train = sutils.to_numeric(sequences, aa2idx)
    x_train = to_categorical(x_train, num_classes=len(sutils.ORDER_LIST))
    print(x_train.shape)
    
    train_hyperparams = dict()
    train_hyperparams["latent_size"] = params.latentdim
    train_hyperparams["steps"] = params.niter
    train_hyperparams["batch_size"] = params.batch_size
    train_hyperparams["lr"] = params.learning_rate
    
    with open(os.path.join(params.output, "hyperparams.json"), "w") as outf:
        json.dump(train_hyperparams, outf)

    model = AAE(x_train, train_hyperparams)
    
    model.build() 
    
    to_save = model.train(x_train, None, params.output, 
                          idx2aa, train_hyperparams["steps"], 
                          train_weights=weights,
                          save_interval=params.save_interval, 
                          to_sample=params.to_sample)
    if to_save:
        model.save(params.output)
    
    sys.exit(0)
    
if __name__ == '__main__':
    params = get_cmd()
    
    if not os.path.isdir(params.output):
        os.makedirs(params.output)
    
    logdir = os.path.join(params.output, "logs")
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    sampledir = os.path.join(params.output, "samples")
    if not os.path.isdir(sampledir):
        os.makedirs(sampledir)

    main(params)
    
