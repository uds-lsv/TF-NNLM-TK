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

import time

import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

from utils import concat_overlap


def data_type():
  return tf.float32
        

class LM():
    """
    This classe implements the basic RNN-LMs using the built-in Tensorflow cells. 
    In particular, this calss can be used to train vanilla-RNN, LSTM (with and 
    witout projection) and GRU.
    """
    
    def __init__(self, config, training=True):
        """ 
        The constructor of the RNN-LM. We define here the complete graph.
        """
        
        # store the configuration for the future
        self.config = config
        
        # define the particular attributes of the basic RNN models
        self.training = training

        # bottleneck layer activation function
        if config.activation == 'tanh':
            self.activation = tf.nn.tanh 
        elif config.activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        elif config.activation == 'relu': 
            self.activation = tf.nn.relu 
        elif config.activation == 'elu':
            self.activation = tf.nn.elu   
        elif config.activation == 'relu6':
            self.activation = tf.nn.relu6 

        self.history_size=1
        self.model = config.model
        self.init_method = config.init_method
        self.num_layers = config.num_layers
        self.input_keep_prob = config.input_keep_prob
        self.output_keep_prob = config.output_keep_prob
        self.vocab_size = config.vocab_size
        self.use_peepholes = config.use_peepholes
        
        # check consistencies in the LSTM parameters
        if config.model == "lstm" or config.model == "lstmp":
            if config.use_peepholes == True or config.lstmp_proj_size:
                self.model = "lstmp"
            else:
                self.model = "lstm"

        if self.model == "lstmp" and config.bottleneck_size:
            print("[WARNING] you are using a bottleneck layer on the the top of an LSTMP model, which includes an internal bottleneck (projection) layer...!")
        
        self.embed_size = config.embed_size
        self.hidden_state_size = config.hidden_size    # hidden size (layer): internal to the models (e.g., memory in LSTM).
        self.recurrent_state_size = config.hidden_size # recurrent layer: layer that feeds back in time into the model.
        self.last_layer_size = config.hidden_size      # last layer: layer right before the output layer (can be bottleneck or recurrent layer).

        if config.bottleneck_size: 
            self.last_layer_size = config.bottleneck_size   
                 
        if self.model == "lstmp" and config.lstmp_proj_size:
            self.recurrent_state_size = config.lstmp_proj_size
            self.last_layer_size = config.lstmp_proj_size
            if config.bottleneck_size: 
                self.last_layer_size = config.bottleneck_size  
                

        ############################################################### 
        ##############      DEFINE THE PLACEHOLDERS      ##############
        # placeholder for the training input data and target words
        
        self.input_data = tf.placeholder(
            tf.int32, [config.batch_size, config.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [config.batch_size, config.seq_length])
        
        
        ############################################################### 
        #######  DEFINE TRAINABLE VARIABLES (WEIGHTS AND BIASES)  #####

        # define the initializer of embeddings, weights and biases
        if self.init_method == "xavier": 
            initializer = tf.contrib.layers.xavier_initializer(uniform=True) 
        else: 
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            
        # word embeddings
        with tf.variable_scope("input_layer"):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embed_size], 
                                        initializer=initializer)

        # weights and biases of the bottleneck layer (if used) 
        if config.bottleneck_size:
            with tf.variable_scope("bottleneck_layer"):
                self.bottleneck_w = tf.get_variable("bottleneck_w", 
                                                [self.recurrent_state_size, config.bottleneck_size],
                                                initializer=initializer)
                self.bottleneck_b = tf.get_variable("bottleneck_b", [config.bottleneck_size], 
                                                    initializer=initializer) 
                
        # weights and biases of the hidden-to-output layer
        with tf.variable_scope("output_layer"):
            self.output_w = tf.get_variable("output_w", 
                                            [self.last_layer_size, self.vocab_size], 
                                            initializer=initializer)
            self.output_b = tf.get_variable("output_b", [self.vocab_size], 
                                            initializer=initializer) 
      
                
        ############################################################### 
        ##########         BUILD THE LM NETWORK GRAPH        ##########
                          
        # extract the embedding of each char input in the batch
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)

        # apply dropout to the input if needed.
        if self.training and self.input_keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self.input_keep_prob)

        # rearrange our input shape to create the training sequence
        # we create a sequence made of the vertical slices the input
        inputs = tf.split(inputs, config.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # build the separate forward and backward lstm cells
        self.cell = self.build_recurrence_cell(config)
    
        # initialize the hidden (recurrent) state to zero
        self.initial_state = self.cell.zero_state(config.batch_size, tf.float32)
        
        # build the LM and update the hidden state
        rec_state, self.final_state = self.time_sequence_graph(inputs)
     
        if config.bottleneck_size:
            last_layer = self.activation(tf.nn.xw_plus_b(rec_state, self.bottleneck_w, self.bottleneck_b))
        else:
            last_layer = rec_state 
            
        #self.logits = tf.matmul(output, self.output_w) + self.output_b
        #self.probs = tf.nn.softmax(self.logits)
        logits = tf.nn.xw_plus_b(last_layer, self.output_w, self.output_b)
        # reshape logits to be a 3-D tensor for sequence loss
        self.logits = tf.reshape(logits, [config.batch_size, config.seq_length, self.vocab_size])
        
        loss = tf.contrib.seq2seq.sequence_loss(
            self.logits,
            self.targets,
            tf.ones([config.batch_size, config.seq_length], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)
    
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss)        

        ###################################################
        # If we are in the training stage, then  calculate the loss, back-propagate
        # the error and update the weights, biases and word embeddings 
        
        if self.training:
            self.lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            # clip the gradient by norm
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.grad_clip)
            # update variables (weights, biases, embeddings...)
            with tf.name_scope('optimizer'):
                #optimizer = tf.train.AdamOptimizer(self.lr)
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                    global_step=tf.contrib.framework.get_or_create_global_step())
    


    def build_recurrence_cell(self, config): 
        """
        Build and return the recurrent cell that will be used by our LM.
        This class uses only the built-in Tensorflow 
        """
        # if needed, the activation function used by the basic model can change be changed as well
        activation_ = tf.nn.tanh
        
        if self.model == 'rnn':
            _cell_ = rnn.BasicRNNCell
        elif self.model == 'gru':
            _cell_ = rnn.GRUCell
        elif self.model == "lstmp":
            _cell_ = rnn.LSTMCell    
        elif self.model == "lstm":
            _cell_ = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(self.model))

        cells = []
        
        # Apply dropout if required
        for _ in range(self.num_layers):
            if  self.model == "lstmp": # you can change the activation function of the project layer
                cell = _cell_(self.hidden_state_size, use_peepholes=self.use_peepholes, num_proj=config.lstmp_proj_size)
            else:    
                cell = _cell_(self.hidden_state_size, activation=activation_)
                
            if self.training and self.output_keep_prob < 1.0 :
                cell = rnn.DropoutWrapper(cell,output_keep_prob=self.output_keep_prob)
            cells.append(cell)

        # build and return the TF multi-recurrent cell graph
        return rnn.MultiRNNCell(cells, state_is_tuple=True)
   

        
    def time_sequence_graph(self, inputs):
        """
        Apply the recurrence cell to an input sequence (each batch entry is a sequence of words).
        return: stacked cell outputs of the complete sequence in addition to the last hidden state 
        (and memory for LSTM/LSTMP) obtained after processing the last word (in each batch entry).
        """
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.cell, loop_function=None)
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.recurrent_state_size])
            
        return output, last_state
    

    def run_model(self, session, data, eval_op=None, verbosity=10000, verbose=False):
        """
        Train or test the current model on some given data.
        This basically trains/applies the model on some data 
        loaded by the data processor.  
        This will help training on a large corpus by splitting 
        them into smaller chunks and processing them one by one. 
        """
        
        data.reset_batch_pointer()
        
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = session.run(self.initial_state)

        fetches = {
            "cost": self.cost,
            "final_state": self.final_state,
        }
        if eval_op is not None:
            fetches["eval_op"] = eval_op

        print_tresh = 0 
        for step in range(data.num_batches):
            input, target = data.next_batch()         
            feed_dict = {self.initial_state: state, self.input_data: input, self.targets: target}
            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]

            costs += cost
            iters += data.seq_length
            
            total_proc_words = float((iters-1)*data.batch_size)  

            if verbose and (step==0 or total_proc_words > print_tresh or step == data.num_batches-1) :      
                print("[INFO] Progress: {:.2f}% | Perplexity: {:.3f} | Total Words: {:.1f}K | Speed: {:.1f}K word/second".format(
                    (step+1) / (data.num_batches) * 100, np.exp(costs/iters), 
                    total_proc_words / 1000,
                    total_proc_words / (1000 *(time.time() - start_time))))
                print_tresh += verbosity
                        
        return np.exp(costs / iters)
    
