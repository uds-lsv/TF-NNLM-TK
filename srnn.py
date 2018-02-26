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

import time

import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

from utils import concat_overlap, xewy_plus_z, xscalary_plus_z


def data_type():
  return tf.float32
        

class LM():
    """
    LM to predict the next word in a sequence given its context.
    """
    
    def __init__(self, config, training=True):
        """ 
        The constructor of the SRNN-LM. We define here the complete graph.
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

        self.model = config.model
        self.init_method = config.init_method

        self.history_size = config.history_size
        self.input_len = config.seq_length + config.history_size - 1 
        self.input_keep_prob = config.input_keep_prob
        self.srnn_keep_prob = config.srnn_keep_prob
        self.output_keep_prob = config.output_keep_prob
        
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.embed_size = config.embed_size
        self.hidden_size = config.hidden_size
        self.recurrent_state_size = config.embed_size
        
        print("[INFO] Initializing model: {}".format(self.model))  
        if config.num_layers <= 0 :
            print("\n[WARNING] number of non-recurrent hidden layers in SRNN models cannot be 0 ===> --num_layers should be > 0 \n")
            exit(0)
            
        ############################################################### 
        ##############      DEFINE THE PLACEHOLDERS      ##############
        # placeholder for the training input data and target words
        
        self.input_data = tf.placeholder(
            tf.int32, [config.batch_size, self.input_len])
        self.targets = tf.placeholder(
            tf.int32, [config.batch_size, config.seq_length])
                
        ############################################################### 
        #######  DEFINE TRAINABLE VARIABLES (WEIGHTS AND BIASES)  #####
        
        # Define the initializer of embeddings, weights and biases
        if self.init_method == "xavier": 
            initializer = tf.contrib.layers.xavier_initializer(uniform=False) 
        else: 
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        # variable for word embeddings
        with tf.variable_scope("input_layer"):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embed_size], 
                                        initializer=initializer)

        # define and initialize the SRNN weight vector/matrix.
        wisrnn_init_vector = np.random.uniform(0.0, 1.0, self.recurrent_state_size)
        # WD-SRNN
        if self.model == "wd-srnn":
            with tf.variable_scope("srnn_layer", reuse=False):
                # for WDSRNN, we initialize all sequential embeddings to the same random vector between 0 and 1.
                wdsrnn_init_matrix=np.array([wisrnn_init_vector,]*self.vocab_size)
                wdsrnn_initializer = tf.constant_initializer(wdsrnn_init_matrix)
                srnn_embedding = tf.get_variable("srnn_embedding", [self.vocab_size, self.recurrent_state_size],
                                                 initializer=wdsrnn_initializer)

        # WI-SRNN
        if self.model == "wi-srnn" :
            with tf.variable_scope("srnn_layer", reuse=False):
                wisrnn_initializer = tf.constant_initializer(wisrnn_init_vector)
                srnn_weight_vector = tf.get_variable("srnn_weight_vector",  self.recurrent_state_size, 
                                                     initializer=wisrnn_initializer)

        # FF-SRNN (forgetting factor)
        if self.model == "ff-srnn" :
            with tf.variable_scope("srnn_layer", reuse=False):
                self.srnn_forgetting_factor = tf.constant(config.forgetting_factor) 
                
        # weights and biases of non-recurrent layer(s)
        self.hidden_w = [] 
        self.hidden_b = [] 
        with tf.variable_scope("non_recurrent_layers"):
            self.hidden_w.append(tf.get_variable("hidden_w/1", 
                                               [self.recurrent_state_size*self.history_size, self.hidden_size], 
                                               initializer=initializer))
            self.hidden_b.append(tf.get_variable("hidden_b/1", [self.hidden_size], 
                                                 initializer=initializer))

            for layer in range(1, self.num_layers):
                self.hidden_w.append(tf.get_variable("hidden_w/" + str(layer+1), 
                                                   [self.hidden_size, self.hidden_size],
                                                   initializer=initializer))
                self.hidden_b.append(tf.get_variable("hidden_b/" + str(layer+1), [self.hidden_size],
                                                   initializer=initializer)) 
     
        # weights and biases of the hidden-to-output layer
        with tf.variable_scope("output_layer"):
            self.output_w = tf.get_variable("output_w", 
                                            [self.hidden_size, self.vocab_size],
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
        inputs = tf.split(inputs, self.input_len, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]        

        # The next two lines are just a hack to initialize the SRNN cell from
        # the TF built-in RNN cell.
        srnn_cell = rnn.BasicRNNCell(self.embed_size)
        self.initial_state = srnn_cell.zero_state(config.batch_size, tf.float32)
        
        # build the LM and update the hidden state
        last_layer, self.final_state = self.srnn_sequence_graph(config, inputs)

        
        for layer in range(config.num_layers): 
            last_layer = self.activation(tf.nn.xw_plus_b(last_layer, self.hidden_w[layer], self.hidden_b[layer]))
            if self.training and config.output_keep_prob < 1:
                last_layer = tf.nn.dropout(last_layer, config.output_keep_prob)
            
        #self.logits = tf.matmul(output, self.output_w) + self.output_b
        #self.probs = tf.nn.softmax(self.logits)

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
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                                  config.grad_clip)
            # update variables (weights, biases, embeddings...)
            with tf.name_scope('optimizer'):
                #optimizer = tf.train.AdamOptimizer(self.lr)
                #self.train_op = optimizer.apply_gradients(zip(grads, tvars))
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
                self.train_op = optimizer.apply_gradients(
                    zip(grads, tvars),
                    global_step=tf.contrib.framework.get_or_create_global_step())


    def srnn_sequence_graph(self, config, inputs):
        """
        Build the recurrence graph of the LSRC model. 
        It returns a list of the hidden outputs and the last hidden layer
        """
        outputs = []
        state = self.initial_state
        with tf.variable_scope("srnn_layer", reuse=True):
            
            if self.model == "wi-srnn":
                wi_srnn_weights = tf.get_variable("srnn_weight_vector")
            
                for iter in range(config.seq_length+self.history_size-1):
                    state = xewy_plus_z(wi_srnn_weights, state, inputs[iter], tf.nn.tanh)
                    #state = tf.multiply(wi_srnn_weights, state) 
                    #state = tf.nn.tanh(tf.add(state, inputs[iter]))
                    outputs.append(state)

            elif self.model == "wd-srnn":
                srnn_embedding = tf.get_variable("srnn_embedding")
                wd_srnn_weights = tf.nn.embedding_lookup(srnn_embedding, self.input_data)
                wd_srnn_weights = tf.split(wd_srnn_weights, self.input_len, 1)
                wd_srnn_weights = [tf.squeeze(srnn_embed_, [1]) for srnn_embed_ in wd_srnn_weights]  
            
                for iter in range(config.seq_length+self.history_size-1):
                    state = xewy_plus_z(wd_srnn_weights[iter], state, inputs[iter], tf.nn.tanh)
                    #state = tf.multiply(wd_srnn_weights[iter], state) 
                    #state = tf.nn.tanh(tf.add(state, inputs[iter]))
                    outputs.append(state)

            elif self.model == "ff-srnn":            
                for iter in range(config.seq_length+self.history_size-1):
                    state = xscalary_plus_z(self.srnn_forgetting_factor, state, inputs[iter], tf.nn.tanh)
                    #state = tf.scalar_mul(self.srnn_forgetting_factor, state) 
                    #state = tf.nn.tanh(tf.add(state, inputs[iter]))
                    outputs.append(state)
                                      
            last_state = outputs[-self.history_size]
            output_ = concat_overlap(outputs, self.history_size, (self.history_size-1), concat_axis=1)
            output = tf.reshape(tf.concat(output_, 1), [-1, self.recurrent_state_size*self.history_size])
            
            # apply dropout to the input if required.
            if self.training and self.srnn_keep_prob < 1:
                output = tf.nn.dropout(output, self.srnn_keep_prob)

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
    
