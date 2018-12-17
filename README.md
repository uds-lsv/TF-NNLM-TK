# TF-NNLM-TK: A Neural Network Language Model Toolkit in Tensorflow

## About
TF-NNLM-TK is a toolkit written in Python3 for neural network language modeling using Tensorflow. It includes basic models like RNNs and LSTMs as well as more advanced models. It provides functionality to preprocess the data, train the models and evaluate them. The toolkit is open-source under the Apache 2 license.

Currently, the following models are supported:
- Vanilla-RNN
- LSTM
- LSTM with projection
- GRU
- Sequential RNN (word-dependent, word-independent and with forgetting-factor) 
- Long-Short Range Context

## A First Example
First install Python and Tensorflow. The code is tested with Python 3 and Tensorflow 1.8. 

For this first example, we need to download the toolkit and some training data. We'll use the PTB dataset provided in Tomas Mikolov's tutorial. For this, you can run the following code in your command line:

```bash
git clone https://repos.lsv.uni-saarland.de/youalil/TF-NNLM-TK.git TODO replace
cd TF-NNLM-TK
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz
```
Then, to train and evaluate a first, simple neural language model, just run the following code
```bash
python train_basic_rnn_models.py --save_dir=small_lstm --model=lstm --train_file=simple-examples/data/ptb.train.txt --test_file=simple-examples/data/ptb.test.txt

python test.py --model_file=small_lstm/model.ckpt --test_file=simple-examples/data/ptb.test.txt
```
The training takes about 20 minutes on a GTX 1050Ti GPU.

## The Details

### Data Preprocessing:
*The training scripts already call the data processing code. If you use the default data, you can probably skip this section.*

The toolkit includes a data processor, which reads a text file and creates from it
two (numpy) arrays that store the batches of input words (history) and target words (prediction).
The code also provides a few helpful tools such as functionality to create and save the vocabulary, to create counts or to map OOV words in test files into a given UNKNOWN token. 

This part of the code does not have a main function. Instead, it is directly called in the Python code of the training scripts. For example, you can call in Python
```python
DataProcessor(train_file, batch_size, seq_length, True, '<unk>', history_size=history_size)
```
This code snippet will take the train_file, do the transformations described above, and create batches from it using the given parameters. 

### Model Description

The TF-NNLM-TK provides the training code for the following neural language models:

#### Basic RNN models
These are commonly known and used. In particular, this toolkit implements Vanilla-RNN, LSTM, 
LSTM with projection and GRU. These models can be trained using the script **train_basic_rnn_models.py** (see example below).

#### Sequential RNN models
These models use more than N words from the history instead of the last one. The implementation provides three models: Word-Dependent SRNN (WD-SRNN), Word-Independent SRNN (WI-SRNN) and Forgetting-Factor SRNN (FF-SRNN). More information about these models can be found [here](http://www.isca-speech.org/archive/Interspeech_2016/pdfs/0422.PDF). These models can be trained using the script **train_srnn.py**

#### Long-Short Range Context (LSRC) model
These models use two separate local and global states to learn short and long-range dependencies separately. The segmental TF implementation of back-propagation causes this model to drastically suffer from the vanishing gradient in the local state, which uses a Vanilla-RNN model, thus the latter is replaced (temporarily by a GRU). More information about this model can be found [here](http://www.aclweb.org/anthology/D16-1154). This model can be trained using the script **train_lsrc.py** 

### Training

Each of these training scripts (train_basic_rnn_models.py, train_srnn.py and train_lsrc.py) includes a large number of parameters, each of them has a description attached to it. To obtain this description run, for example, on your command line: 

```bash
python train_basic_rnn_models.py --help
```
The default parameters of all models try to match the small configuration reported in the [Tensorflow PTB-LM](https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py) recipe:

| config | epochs | train | valid  | test
|--------|--------|-------|--------|-------
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29

To reproduce these numbers with the LSTM model (actually better numbers due to Xavier initialization), run (adjusting the path to the data to your setting):
```bash
python train_basic_rnn_models.py --save_dir=small_lstm --model=lstm --train_file=path/to/data/train.txt --test_file=path/to/data/test.txt
```

This call will train the LSTM model on the PTB data using the exact same configuration used in the Tensorflow recipe. If you want to run the model with the medium configuration, you will simply need to set the parameters to their values specified in the medium configuration:

```bash
python train_basic_rnn_models.py --init_scale=0.05 --seq_length=35 --embed_size=650 --hidden_size=650 --max_epoch=6 --num_epochs=39 --decay_rate=0.8 --batch_size=20 --input_keep_prob=0.5 --output_keep_prob=0.5 --model=lstm --save_dir=medium_lstm --train_file=path/to/data/train.txt --test_file=path/to/data/test.txt
```

The same idea applies to the more advanced models, except that you need to call the corresponding training scripts and that you might need to slightly adjust the parameters. Try for example: 

```bash
python train_srnn.py --model=wi-srnn --input_keep_prob=0.6 --save_dir=wisrnn_small_5gram --train_file=path/to/data/train.txt --test_file=path/to/data/test.txt 
```

to train the WISRNN model with the default configuration except for the word embedding dropout, which is set to 0.4 (1-0.6). This should lead to a performance of ~109.5 on the example dataset from above.

Similarly, the LSRC model is trained using the corresponding script:

```bash
python train_lsrc.py --save_dir=lsrc_small --train_file=path/to/data/train.txt --test_file=path/to/data/test.txt 
```

The script also allows modifying the training by setting the corresponding parameters. Use --help to get more information about them.

### Testing

The test script is the same for all models. You only need to specify the path to the model
that you want to evaluate and the path to your test file. To evaluate the small LSTM model we trained above, we just need to run the following command: 
```bash
python test.py --model_file=small_lstm/model.ckpt --test_file=path/to/data/test.txt 
```
The script also offers a few other parameters to control the speed in case you are testing on a very large corpus.

## Authors and Licence

This toolkit was originally developed by Youssef Oualil, during his time at [LSV, Saarland University](https://www.lsv.uni-saarland.de/). It is currently mantained at the LSV group by [Michael A. Hedderich](https://www.lsv.uni-saarland.de/?id=188) with contributions by Adam Kusmirek. This work was funded in part by SFB 1102. 

This code is licensed under Apache 2.0. Parts of this code are based on the Tensorflow PTB-LM recipe licensed under the Apache License, Version 2.0 by the TensorFlow Authors. Please see the LICENCE file for details. 

If you use this toolkit as part of a publication, please consider citing us:

```
@inproceedings{oualil-singh-greenberg-klakow:EMNLP2016,
  author = {Oualil, Youssef  and  Singh, Mittul  and  Greenberg, Clayton  and  Klakow, Dietrich},
  title = {Long-Short Range Context Neural Networks for Language Modeling},
  booktitle = {{EMNLP} 2016, Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing},
  day = {3},
  month = {November},
  year = {2016},
  address = {Austin, Texas},
  publisher = {Association for Computational Linguistics},
  pages = {1473--1481},
  url = {http://aclweb.org/anthology/D16-1154.pdf},
  poster = {http://coli.uni-saarland.de/~claytong/posters/EMNLP16_Poster.pdf}
}
```

or

```
@inproceedings{oualil-greenberg-singh-klakow:2016:IS,
  author = {Youssef Oualil and Clayton Greenberg and Mittul Singh and Dietrich Klakow},
  title = {Sequential Recurrent Neural Networks for Language Modeling},
  day = {12},
  month = {September},
  year = 2016,
  address = {San Francisco, California, USA},
  booktitle = {{INTERSPEECH} 2016, Proceedings of the 17th Annual Conference of the International Speech Communication Association},
  doi = {10.21437/Interspeech.2016-422},
  url = {http://www.isca-speech.org/archive/Interspeech_2016/pdfs/0422.PDF},
  pages = {3509--3513},
  publisher = {{ISCA}}
}
```

 


