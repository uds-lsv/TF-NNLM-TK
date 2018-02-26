# Copyright 2018 Saarland University, Spoken Language
# Systems LSV (author: Youssef Oualil)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
#
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Parts of this code are based on the Tensorflow PTB-LM recipe licensed under 
# the Apache License, Version 2.0 by the TensorFlow Authors.
# (Source: https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
# retrieved in January 2018)
############################################################################### 

from __future__ import print_function

import time
import sys
import os

import copy
import argparse

from six.moves import cPickle
import tensorflow as tf
import numpy as np

from data_processor import DataProcessor
from lsrc import LM

     
def main():
    # we define some parameters that can be set by the user
    # read description of each argument for more info
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--train_file', type=str, default='data/ptb/ptb.train.txt',
                        help='path to the training file containing one sentence per line (</eos> is automatically added).')
    parser.add_argument('--test_file', type=str, default='data/ptb/ptb.test.txt',
                        help='path to the test (or validation) file containing one sentence per line (</eos> is automatically added).')
    parser.add_argument('--save_dir', type=str, default='lsrc_medium',
                        help='directory to store intermediate models, model configuration and final model')

    parser.add_argument('--embed_size', type=int, default=650,
                        help='size of the word embeddings')
    parser.add_argument('--hidden_size', type=int, default=650,
                        help='size of the (recurrent) hidden layers')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of recurrent layers')
    parser.add_argument('--bottleneck_size', type=int, default=None,
                        help='number of neurons in the additional bottleneck layer for recurrent models (None = not used)')
    
    parser.add_argument('--lstmp_proj_size', type=int, default=None,
                        help='number of neurons in the additional projection layer of the LSTMP model (None = not used)')
    parser.add_argument('--use_peepholes', type=bool, default=False,
                        help='use peepholes in the LSTM/LSTMP model (this parameters is used only in LSTM/LSTMP models)')
    parser.add_argument('--activation', type=str, default='relu',
                        help='activation function of the bottleneck layer (if not None)')

    parser.add_argument('--output_keep_prob', type=float, default=0.5,
                        help='probability of keeping activations in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=0.5,
                        help='probability of keeping the inputs in the word embeddings')
    
    parser.add_argument('--batch_size', type=int, default=20,
                        help='mini-batch size')
    parser.add_argument('--seq_length', type=int, default=35,
                        help='word sequence length processed at each forward-backward pass')
    
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('--max_epoch', type=int, default=6,
                        help='number of epoch before we start using learning decay')
    
    parser.add_argument('--init_method', type=str, default='xavier',
                        help='initialization method of embeddings, weights and biases. It can be either xavier or None.\
                        The latter will use the default uniform distribution on the interval given by --init_scale')
    parser.add_argument('--init_scale', type=float, default=0.05,
                        help='interval for initialization of variables. This is used only if init_method = None')
    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.8,
                        help='learning rate decay')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='clip gradients at this value')

    config = parser.parse_args()
    train(config)
  
  
def train(config):
    
    # training and test configuration are basically the same     
    config_test = copy.deepcopy(config)
    config_test.batch_size = 10
    config_test.seq_length = 1
    
    # process the training corpus (if not done yet) and return the training batches and other info 
    train_data = DataProcessor(config.train_file, config.batch_size, config.seq_length, True, '<unk>', history_size=1)
    test_data  = DataProcessor(config_test.test_file, config_test.batch_size, config_test.seq_length, False, '<unk>', history_size=1)
 
    config.vocab_size = train_data.vocab_size
    config_test.vocab_size = train_data.vocab_size
 
     # save the training configuration for future need
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)
    try:
        with open(os.path.join(config.save_dir, 'config.pkl'), 'wb') as f:
            cPickle.dump(config, f)
    except IOError:
        print("ERROR: Could not open and/or write the config file {}".format(
            os.path.join(config.save_dir, 'config.pkl'))) 
        
        
    with tf.Graph().as_default():

        # create the LM graph for training
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None):
                model_train = LM(config, True)
        
        # create the LM graph for testing with shared parameters 
        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True):
                model_test = LM(config_test, False)
                    
        # run  the training/testing
        with tf.Session() as session:
            
            session.run(tf.global_variables_initializer())

            test_perplexity = model_test.run_model(session, test_data, eval_op=None, verbosity=10000, verbose=True)
            print("\n[INFO] Starting perplexity of test set: %.3f" % test_perplexity)
            print('========================\n')
            
            # model saving manager 
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            
            # loop over all epochs       
            for e in range(config.num_epochs):
                
                # we reset/define the epoch parameters
                lr_decay = config.decay_rate ** max(e + 1 - config.max_epoch, 0.0)        
                session.run(tf.assign(model_train.lr, config.learning_rate * lr_decay))
            
                print("[INFO] Epoch: %d, Learning rate: %.3f \n" % (e + 1, session.run(model_train.lr)))
                train_perplexity = model_train.run_model(session, train_data,
                                                         eval_op=model_train.train_op, verbosity=50000, verbose=True)
                
                test_perplexity = model_test.run_model(session, test_data)
                print("\n[SUMMARY] Epoch: {} | Train Perplexity: {:.3f} | Test Perplexity: {:.3f} \n".format(e + 1, train_perplexity, test_perplexity))
                print('========================')
            
                # save model after each epoch        
                model_path = os.path.join(config.save_dir, 'model.ckpt')
                saver.save(session, model_path, global_step=(e+1))                
        
            # save the final model
            model_path = os.path.join(config.save_dir, 'model.ckpt')
            saver.save(session, model_path)   
               
                                       
if __name__ == '__main__':
    main()


