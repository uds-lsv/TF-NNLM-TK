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

import io
import os
import sys
import collections

from six.moves import cPickle
import numpy as np

from utils import chunk


Python_version = sys.version_info[0] == 3


        
class DataProcessor():
    """
    Class to read a corpus and transform it into two (numpy) arrays storing input and target batches. 
    If the corpus has been already processed, the model then loads the stored and processed data.
    """
    
    def __init__(self, corpus_path, batch_size, seq_length, is_training=False, 
                 unk='<unk>', history_size=1, vocab_dir_path=None):
        """
        Constructor: Read a corpus and transforms it into a sequence of batches.
        
        :param corpus_path 
        :param batch_size
        :param seq_length
        :param is_training
        :param unk
        :param history_size
        :param vocab_dir_path: Directory of the vocabulary file. If None,
                then the directory of the corpus will be used.
        """
        # In case another encoding is needed 
        self.encoding = 'utf-8'
        
        self.batch_size  = batch_size
        self.seq_length  = seq_length
        self.history_size= history_size
        self.data_dir    = os.path.dirname(corpus_path)
        self.is_training = is_training
        self.unk = unk
        
        if vocab_dir_path is None:
            vocab_dir_path = self.data_dir
            
        vocab_file  = os.path.join(vocab_dir_path, "vocabulary.pkl") 
        nparray_file = os.path.join(self.data_dir, os.path.basename(corpus_path)[:-3] +  "npy")
            
        if not os.path.exists(vocab_file) and not self.is_training:
            raise Exception("Vocabulary file {} does not exist but must exist for prediction.".format(vocab_file))
            
        if not (os.path.exists(vocab_file) and os.path.exists(nparray_file)):
            print("Processing raw data file {}...!".format(corpus_path))
            self.process_file(corpus_path, vocab_dir_path, nparray_file)
        else:
            print("Vocab file {} and processed corpus file {} already exist from past run, loading...!".format(
                        vocab_file, nparray_file))
            self.load_saved_vocabulary(vocab_file)
            self.load_saved_corpus(nparray_file)
        
        # create the input/target batches from the data array 
        self.data_to_batches()  

        # make sure that the index points at the first batch 
        self.reset_batch_pointer()


    def process_file(self, input_file, vocab_dir_path, nparray_file):
        """
        Read and process corpus from file and save it as a numpy array, in addition to saving the vocabulary as well.
        """  
        word_list = self._file_to_word_list(input_file)
        
        # If we are in the training phase, we need to create the vocabulary as well
        if self.is_training: 
            self.vocab, _, count_pairs = self._create_vocab_from_list(word_list)
            count_string = self._dict_to_string(dict(count_pairs)) 
            self._save_file(os.path.join(self.data_dir, "counts.txt"), count_string, 'text')
            self.vocab_size = len(self.vocab)
            self.words = sorted(tuple(self.vocab.keys()))
        
            self.save_vocabulary(vocab_dir_path)
        
        # If we are in the test phase, we need to map all OOV words into UNK symbol
        else: 
            _, file_words, _ = self._create_vocab_from_list(word_list)
            self.vocab = self._read_pickle_file(os.path.join(vocab_dir_path, "vocabulary.pkl"))
            
            train_words = set(self.vocab.keys())
            word_list = [word if word in train_words else self.unk for word in word_list]
        
        # create an array of our mapped data: each word is replaced by its ID 
        self.data = np.array(list(map(self.vocab.get, word_list)))
        self._save_file(nparray_file, self.data, 'numpy')

    def _read_text_file(self, filename):
        """ 
        Read a text file and return its content
        """              
        try:
            with open(filename, "r", encoding=self.encoding) as f:
                return f.read()  
        except IOError:
                raise Exception("ERROR: Could not open and/or read file {}".format(filename))


    def _read_pickle_file(self, filename):
        """ 
        Read a pickle file and return its content
        """
        try:
            with open(filename, 'rb') as f:
                return cPickle.load(f)
        except IOError: 
            raise Exception("Could not open and/or read pickle file {}".format(filename))  
            
                           
    def _save_file(self, filename, content, filetype='text'):
        """ 
        Save some content into a file for three possible formats: text, numpy and pickle.  
        """
        try:
            if filetype == 'numpy':
                    np.save(filename,content)
            elif filetype == 'text':
                with io.open(filename, 'w') as f:
                    f.write(content)
            elif filetype == 'pickle':
                with io.open(filename, 'wb') as f:
                    cPickle.dump(content, f)
            else:
                raise Exception("File type {} unknown.".format(filetype))   
        except IOError:
            raise Exception("Could not write and/or save file {}".format(filename))
     

    def _dict_to_string(self, dictionary):
        """
        Turn a list of words into a long string. This is useful for debugging.
        """
        list_dict = [ str(d)+'\t'+str(dictionary[d]) for d in dictionary ]
        return '\n'.join(list_dict)
    
                   
    def _file_to_word_list(self, filename):
        """ 
        Read a file and return its content as a list with newline replaced by <eps> tag.
        """
        content = self._read_text_file(filename)
        if Python_version:
            return content.replace("\n", "<eos>").split()
        else:
            return content.decode("utf-8").replace("\n", "<eos>").split()


    def _create_vocab_from_list(self, word_list):
        """ 
        Create the vocabulary map (a dictionary) from a list of words (read from a file).
        We also return the list of unique words in addition to their counts.
        """
        counter = collections.Counter(word_list)
        count_pairs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        words, _ = zip(*count_pairs)
        # Create our vocabulary map: dict{..., word:ID, ...} 
        vocab = dict(zip(words, range(len(words))))
        return vocab, words, count_pairs
    
    
    def load_saved_vocabulary(self, vocab_file):
        """ 
        Load a previously created vocabulary file
        """
        try:
            with open(vocab_file, 'rb') as f:
                self.vocab = cPickle.load(f)
        except IOError: 
            raise Exception("Could not open and/or read pickle file {}".format(vocab_file))   
            
        self.words = sorted(tuple(self.vocab.keys()))
        self.vocab_size = len(self.words)    
    
    def load_saved_corpus(self, nparray_file):
        """
        Load a previously processed corpus from a file storing the data as a numpy array.
        """
        
        try:
            self.data = np.load(nparray_file)
        except IOError:  
            raise Exception("Could not open and/or read  data (numpy array) file {}".format(nparray_file))  
        self.num_batches = int(self.data.size / (self.batch_size *
                                                   self.seq_length))

    def save_vocabulary(self, save_dir_path):
        """
        Saves the vocabulary to the soecified directory in form of a 
        human-readbale vocabulary.txt and a pickled vocabular.pkl.
        """
        vocab_string = self._dict_to_string(self.vocab) 
        self._save_file(os.path.join(save_dir_path, "vocabulary.txt"), vocab_string, 'text')
        
        self._save_file(os.path.join(save_dir_path, "vocabulary.pkl") , self.vocab, 'pickle')

    def data_to_batches(self):
        """
        Create batches from data stored in an (numpy) array.
        """
        
        self.num_batches = int((self.data.size - self.history_size ) / (self.batch_size *
                                                   self.seq_length))
        
        # Print an error message when the data array is too small
        if self.num_batches == 0:
            assert False, "ERROR: Cannot create batches ==> data size={}, \
             batch size={}, segment size={}".format(self.data.size, 
             self.batch_size, self.seq_length) 

        self.data = self.data[:(self.num_batches * self.batch_size * self.seq_length) + self.history_size ]
        
        # Remove the last words in the input chunk and shift the target words
        input = self.data[:-1]
        target = np.copy(self.data)
        target = target[self.history_size:]

        input = np.array(chunk(input, (self.num_batches*self.seq_length) + self.history_size-1, overlap=self.history_size-1))
        target = np.array(chunk(target, (self.num_batches*self.seq_length), overlap=0))
        
        self.input  = chunk(input, self.seq_length + self.history_size-1, overlap=self.history_size-1) 
        self.target = chunk(target, self.seq_length, overlap=0) 


    def next_batch(self):
        """
        Move to the next batch, this is needed to process the batches in a sequence.
        """
        x, y = self.input[self.pointer], self.target[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        """
        Reset the batch pointer to the beginning of the data.
        """
        self.pointer = 0        

        



         
