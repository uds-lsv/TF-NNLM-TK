# Copyright 2018 Saarland University, Spoken Language
# Systems LSV (author: Youssef Oualil, during his work period at LSV)
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

from utils import xewy_plus_z


def data_type():
    return tf.float32
        

# Not used
# LSRCTuple = collections.namedtuple("LSRC", ("Local", "Global"))


class LM(object):
    """
    This classe implements the LSRC model. 
    """
    
    def __init__(self, config, training=True):
        """ 
        The constructor of the LSRC-LM. We define here the complete graph.
        """
        
        # store the configuration for the future
        self.config = config
        
        # define the attributes of the LSRC model
        self.training = training
        
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
    
        self.model = 'lsrc'
        self.history_size = 1

        self.init_method = config.init_method
        self.num_layers = config.num_layers
        self.input_keep_prob = config.input_keep_prob
        self.output_keep_prob = config.output_keep_prob
        
        self.embed_size = config.embed_size
        self.bottleneck_size = config.bottleneck_size
        self.local_state_size = config.embed_size
        self.global_state_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        self.use_lstmp = (config.lstmp_proj_size or config.use_peepholes)
            
        ############################################################### 
        # #############      DEFINE THE PLACEHOLDERS      ##############
        # placeholder for the training input data and target words
        
        self.input_data = tf.placeholder(
            tf.int32, [config.batch_size, config.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [config.batch_size, config.seq_length])
                
        ############################################################### 
        # ######  DEFINE TRAINABLE VARIABLES (WEIGHTS AND BIASES)  #####
        
        # define the initializer of embeddings, weights and biases
        if self.init_method == "xavier": 
            initializer = tf.contrib.layers.xavier_initializer(uniform=True) 
        else: 
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        # word embeddings
        with tf.variable_scope("input_layer"):
            self.embedding = tf.get_variable("embedding", [config.vocab_size, config.embed_size], 
                                             initializer=initializer)

        local_weight_init = np.random.uniform(0.0, 1.0, self.embed_size)
        with tf.variable_scope("lsrc_layer"):
            local_initializer = tf.constant_initializer(local_weight_init)
            local_weight = tf.get_variable("lsrc_local_weights",  self.embed_size, 
                                           initializer=local_initializer)

        # weights and biases of the bottleneck layer (if used)
        last_layer_size = self.global_state_size            
        if self.bottleneck_size:
            last_layer_size = self.bottleneck_size
            with tf.variable_scope("bottleneck_layer"):
                self.bottleneck_w = tf.get_variable("bottleneck_w", 
                                                    [self.global_state_size, self.bottleneck_size],
                                                    initializer=initializer)
                self.bottleneck_b = tf.get_variable("bottleneck_b", [self.bottleneck_size],
                                                    initializer=initializer)
     
        # weights and biases of the hidden-to-output layer
        with tf.variable_scope("output_layer"):
            self.output_w = tf.get_variable("output_w", 
                                            [last_layer_size, self.vocab_size],
                                            initializer=initializer)
            self.output_b = tf.get_variable("output_b", [self.vocab_size],
                                            initializer=initializer) 

        ############################################################### 
        # #########         BUILD THE LM NETWORK GRAPH        ##########
                          
        # extract the embedding of each char input in the batch
        inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
    
        # apply dropout to the input if needed.
        if self.training and self.input_keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self.input_keep_prob)

        # rearrange our input shape to create the training sequence
        # we create a sequence made of the vertical slices the input
        inputs = tf.split(inputs, config.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]        

        self.global_cell = self.build_lsrc_global_cells(config)
       
        # initialize all LSRC states to zero
        # The next two lines are just a hack to initialize the SRNN cell from
        # the TF built-in RNN cell.
        local_cell = rnn.BasicRNNCell(self.embed_size)
        self.local_state = local_cell.zero_state(config.batch_size, tf.float32)
        self.global_state = self.global_cell.zero_state(config.batch_size, tf.float32)

        # build the LM and update the hidden state
        rec_local_state, self.final_local_state = self.lsrc_local_sequence_graph(config, inputs)
         
        # build the LM and update the hidden state
        rec_global_state, self.final_global_state = self.lsrc_global_sequence_graph(config, rec_local_state)

        # apply bottleneck if used
        if self.bottleneck_size:
            last_layer = self.activation(tf.nn.xw_plus_b(rec_global_state, self.bottleneck_w, self.bottleneck_b))
        else:
            last_layer = rec_global_state 
            
        # self.logits = tf.matmul(output, self.output_w) + self.output_b
        # self.probs = tf.nn.softmax(self.logits)
        logits = tf.nn.xw_plus_b(last_layer, self.output_w, self.output_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        self.logits = tf.reshape(logits, [config.batch_size, config.seq_length, config.vocab_size])
        
        loss = tf.contrib.seq2seq.sequence_loss(
            self.logits,
            self.targets,
            tf.ones([config.batch_size, config.seq_length], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)
    
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss)        

        ###################################################
        # Training stage 
        # If we are in the training stage, then  calculate the loss, back-propagate
        # the error and update the weights, biases and word embeddings 
        
        if self.training:
            self.lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
                
            # perform gradient clipping
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.grad_clip)
            # update variables (weights, biases, embeddings...)
            with tf.name_scope('optimizer'):
                # optimizer = tf.train.AdamOptimizer(self.lr)
                # self.train_op = optimizer.apply_gradients(zip(grads, tvars))
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
                self.train_op = optimizer.apply_gradients(
                    zip(grads, tvars),
                    global_step=tf.contrib.framework.get_or_create_global_step())

    def build_lsrc_global_cells(self, config): 
        """
        Build and return the global recurrent cell of the LSRC model.
        """
        
        cells_ = []
        for _ in range(config.num_layers):
            
            # define the global state of the LSRC model at this layer
            if self.use_lstmp:
                global_cell_ = rnn.LSTMCell(self.global_state_size, use_peepholes=config.use_peepholes, 
                                            num_proj=config.lstmp_proj_size)
            else:   
                global_cell_ = rnn.BasicLSTMCell(self.global_state_size)
            
            # apply dropout if specified
            if self.training and config.output_keep_prob < 1.0:
                global_cell_ = rnn.DropoutWrapper(global_cell_, output_keep_prob=config.output_keep_prob)
                
            # curren_lsrc_layer = LSRCTuple(local_cell_, global_cell_) 
            cells_.append(global_cell_)
    
            # build and return the recurrent cell graph of LSRC
            return rnn.MultiRNNCell(cells_, state_is_tuple=True)

    def lsrc_local_sequence_graph(self, config, inputs):
        """
        Build the recurrence graph of the local state of the LSRC model. 
        It returns a list of the hidden outputs and the last hidden layer
        """
        outputs = []
        state = self.local_state
        with tf.variable_scope("lsrc_layer", reuse=True):
            lsrc_local_weights = tf.get_variable("lsrc_local_weights")
            activation_ = tf.nn.tanh
            for i in range(config.seq_length):
                state = xewy_plus_z(lsrc_local_weights, state, inputs[i], activation=activation_)
                outputs.append(state)
                
        last_state = outputs[-1]
        # outputs = tf.reshape(tf.concat(outputs, 1), [-1, self.local_state_size])
        
        # apply dropout to the input if required.
        if self.training and self.output_keep_prob < 1:
            outputs = tf.nn.dropout(outputs, self.output_keep_prob)
            outputs = tf.split(outputs, config.seq_length, 0)
            outputs = [tf.squeeze(output_, [0]) for output_ in outputs]
        return outputs, last_state

    def lsrc_global_sequence_graph(self, config, inputs):
        """
        Build the recurrence graph of the global state of the LSRC model. 
        It returns a list of the hidden outputs and the last hidden layer
        """
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.global_state, self.global_cell,
                                                         loop_function=None)
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.global_state_size])
            
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
        local_state = session.run(self.local_state)
        global_state = session.run(self.global_state)

        fetches = {
            "cost": self.cost,
            "final_local_state": self.final_local_state,
            "final_global_state": self.final_global_state,
        }
        if eval_op is not None:
            fetches["eval_op"] = eval_op

        print_tresh = 0 
        for step in range(data.num_batches):
            indata, target = data.next_batch()
            feed_dict = {self.local_state: local_state, self.global_state: global_state, 
                         self.input_data: indata, self.targets: target}
            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            local_state = vals["final_local_state"]
            global_state = vals["final_global_state"]

            costs += cost
            iters += data.seq_length
            
            total_proc_words = float((iters-1)*data.batch_size)  

            if verbose and (step == 0 or total_proc_words > print_tresh or step == data.num_batches-1):
                print("[INFO] Progress: {:.2f}% | "
                      "Perplexity: {:.3f} | "
                      "Total Words: {:.1f}K | "
                      "Speed: {:.1f}K word/second"
                      .format((step+1) / data.num_batches * 100, np.exp(costs/iters),
                              total_proc_words / 1000,
                              total_proc_words / (1000 * (time.time() - start_time))))
                print_tresh += verbosity
                        
        return np.exp(costs / iters)

    #########################################################################################
    # # currently not used, but could be if we do not use the TF built-in local and global cells
    # def initialize_lsrc_cells(self, config):
    #
    #     initial_state = []
    #     for layer in range(self.num_layers):
    #         initial_local_state_ = self.cell[layer].Local.zero_state(config.batch_size, tf.float32)
    #         initial_global_state_ = self.cell[layer].Global.zero_state(config.batch_size, tf.float32)
    #         curren_lsrc_init = LSRCTuple(initial_local_state_, initial_global_state_)
    #         initial_state.append(curren_lsrc_init)
    #     return initial_state

    # # currently not used, but could be if we do not use the TF built-in local and global cells
    # def lsrc_sequence_graph_old(self, config, inputs):
    #     """
    #     Build the recurrence graph of the LSRC model.
    #     It returns the output and the last hidden layer
    #     """
    #
    #     outputs = inputs
    #     last_state = []
    #
    #     for layer in range(self.num_layers):
    #         inputs, last_local_state_ = legacy_seq2seq.rnn_decoder(outputs, self.initial_state[layer].Local,
    #                                                                self.cell[layer].Local, loop_function=None)
    #         outputs, last_global_state_ = legacy_seq2seq.rnn_decoder(inputs, self.initial_state[layer].Global,
    #                                                                  self.cell[layer].Global, loop_function=None)
    #         last_lsrc_states_ = LSRCTuple(last_local_state_, last_global_state_)
    #         last_state.append(last_lsrc_states_)
    #
    #     output = tf.reshape(tf.concat(outputs, 1), [-1, self.global_state_size])
    #
    #     return output, last_state
