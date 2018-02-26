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

from six.moves import cPickle
import argparse

#from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import numpy as np

from data_processor import DataProcessor

from basic_rnn_models import LM as Basic_RNN
from srnn import LM as SRNN
from lsrc import LM as LSRC


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model_file', type=str, default='model.ckpt-15.data-00000-of-00001',
                        help='path to the file storing the model to be evaluated')
    parser.add_argument('--test_file', type=str, default='data/ptb/ptb.test.txt',
                        help='path to the test file containing one sentence per line (</eos> is automatically added)')
    
    parser.add_argument('--batch_size', type=int, default=20,
                        help='mini-batch size')
    parser.add_argument('--seq_length', type=int, default=5,
                        help='word sequence length processed at each forward pass')
    
    test_config = parser.parse_args()
    test_config.save_dir = os.path.dirname(test_config.model_file) 
    
    try:
        with open(os.path.join(test_config.save_dir, 'config.pkl'), 'rb') as f:
            config = cPickle.load(f)
    except IOError:
        raise IOError("ERROR: Could not open and/or read the config file {}.".format(
            os.path.join(test_config.save_dir, 'config.pkl'))) 
    
    # copy the parameters that are specific to the test data
    config.save_dir = test_config.save_dir
    config.test_file = test_config.test_file 
    config.batch_size = test_config.batch_size 
    config.seq_length = test_config.seq_length 
    config.model_path = test_config.model_file 
    
    if not hasattr(config, 'history_size'): 
        config.history_size = 1
    
    calculate_perplexity(config)
    


def calculate_perplexity(config):
    """
    Calculate perplexity for a given (processed) test data 
    """
    
    # load the config files and vocabulary files
    try:  
        with open(os.path.join(config.save_dir, 'vocabulary.pkl'), 'rb') as f:
            words, vocab = cPickle.load(f) 
            word_set = set(words)  
    except IOError:
        print("ERROR: Could not open and/or read the vocabulary file {}".format(
            os.path.join(config.save_dir, 'vocabulary.pkl'))) 

    if not os.path.exists(config.model_path  + '.meta'):
        print("ERROR: Could not open and/or read model file {}".format(config.model_path + '.meta'))
        sys.exit(0)

    # process the test corpus and load it into batches 
    test_data = DataProcessor(config.test_file, config.batch_size, config.seq_length, False, '<unk>', history_size=config.history_size)

    
    # define/load the model  
    with tf.variable_scope("Model", reuse=False):
        if config.model == 'lsrc':
            model_test = LSRC(config, False)
        elif config.model == 'wi-srnn' or config.model == 'wd-srnn'  or config.model == 'ff-srnn' :
            model_test = SRNN(config, False)
        elif config.model == 'lstm' or config.model == 'lstmp' or  config.model == 'rnn' or config.model == 'gru':
            model_test = Basic_RNN(config, False)
        else: 
            raise Exception("model type not supported: {}".format(config.model))

    with tf.Session() as session:
        
        # restore the model
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(session, config.model_path)
        
        test_perplexity = model_test.run_model(session, test_data, eval_op=None, verbosity=10000, verbose=True)
        print("\n[SUMMARY] Perplexity: %.3f" % test_perplexity)
        print('========================\n')
            

    
if __name__ == '__main__':
    main()
    
