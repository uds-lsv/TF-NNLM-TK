# Copyright 2018 Saarland University, Spoken Language
# Systems LSV (author: Youssef Oualil)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS*, WITHOUT WARRANTIES OR CONDITIONS OF ANY
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

import io
import os

import tensorflow as tf
import numpy as np
import math




def chunk(A, seq_len, overlap=0):
    
    if overlap >= seq_len:
        print("ERROR in function chunk: overlap cannot be >= to sequence length")
        exit(0) 
    
    if A.ndim == 1:
        Alen = A.shape[0] 
        n = min(seq_len, Alen-1)
        return [A[i:i+seq_len] for i in range(0, Alen - seq_len+1, seq_len-overlap)]
    elif A.ndim == 2:
        Alen = A.shape[1]
        n = min(seq_len, Alen-1)
        return [A[:,i:i+seq_len] for i in range(0, Alen - seq_len+1, seq_len-overlap)]
    else:
        print("ERROR in function chunk: this function works only for 1-D and 2-D arrays")
        exit(0) 
        
        
def concat_overlap(A, seq_len, overlap=0, concat_axis=0):
    
    if overlap >= seq_len:
        print("ERROR in function chunk: overlap cannot be >= to sequence length")
        exit(0) 
    
    s = A[0].get_shape().as_list()
    r = len(s)
    Alen = len(A)
    n = min(seq_len, Alen-1) 
    return [tf.concat(A[i:i+seq_len], concat_axis) for i in range(0, Alen - seq_len+1, seq_len-overlap)]



def xewy_plus_z(x, y, z, activation=None):
    """
    Element-wise product of two vectors plus a third one, with the application
    of an activation function when it is specified.
    """
    R = tf.add(tf.multiply(x, y), z)
    if activation:
        return activation(R)
    else: 
        return R


def xscalary_plus_z(scalar, x, y, activation=None):
    """
    Multiply a vector by a scalar and add it to a second vector, with the application
    of an activation function when it is specified.
    """
    R = tf.add(tf.scalar_mul(scalar,x), y)
    if activation:
        return activation(R)
    else: 
        return R
    
    
# re-implementation of Xavier initializer for tests
def normalized_initializer(layer_i, layer_i_plus_1):
    range =   0.5*math.sqrt(2/(layer_i + layer_i_plus_1))
    return tf.random_uniform_initializer(-range, range)  


    
