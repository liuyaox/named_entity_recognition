# -*- coding: utf-8 -*-
"""
Created:    2019-11-16 13:36:14
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Input, Embedding, LSTM, Bidirectional, Dropout, Masking
from keras.models import Model
from keras_contrib.layers import CRF


class NerBiLSTM:
    """
    Model: Random Embedding -> BiLSTM -> CRF   无Masking，有mask_zero=True
    """
    def __init__(self, config, rnn_units=128):
        self.name = 'NerBiLSTM'
        self.config = config
        self.RNN_UNITS = rnn_units if rnn_units else config.RNN_UNITS
        self.create_model()
        
        
    def create_model(self):
        inputs = Input(shape=(self.config.MAXLEN, ), dtype='int32', name='inputs')  # (, MAXLEN)
        X = Embedding(self.config.VOCAB_SIZE, self.config.WORD_EMBED_DIM, mask_zero=True)(inputs)   # (, MAXLEN, WORD_EMBED_DIM)
        X = Bidirectional(LSTM(self.RNN_UNITS // 2, return_sequences=True))(X)      # (, MAXLEN, RNN_UNITS)
        X = Dropout(0.3)(X)
        out = CRF(self.config.N_TAGS, sparse_target=True, name='crf')(X)            # (, MAXLEN, N_TAGS)
        self.model = Model(inputs=inputs, outputs=out)
        self.model.summary()



class NerBiLSTM2:
    """
    Model: Random Embedding -> BiLSTM -> CRF   有Masking，无mask_zero=True
    """
    def __init__(self, config, rnn_units=128):
        self.name = 'NerBiLSTM'
        self.config = config
        self.RNN_UNITS = rnn_units if rnn_units else config.RNN_UNITS
        self.create_model()
        
    def create_model(self):
        inputs = Input(shape=(self.config.MAXLEN, ), dtype='int32', name='inputs')  # (, MAXLEN)
        X = Masking(mask_value=0)(inputs)                                           # TODO 计算accuracy时，为什么没有发挥mask作用？
        X = Embedding(self.config.VOCAB_SIZE, self.config.WORD_EMBED_DIM)(X)        # (, MAXLEN, WORD_EMBED_DIM)
        X = Bidirectional(LSTM(self.RNN_UNITS // 2, return_sequences=True))(X)      # (, MAXLEN, RNN_UNITS)
        X = Dropout(0.3)(X)
        out = CRF(self.config.N_TAGS, sparse_target=True, name='crf')(X)            # (, MAXLEN, N_TAGS)
        self.model = Model(inputs=inputs, outputs=out)
        self.model.summary()



class NerBiLSTM3:
    """
    Model: Random Embedding -> BiLSTM -> CRF   无Masking，无mask_zero=True
    """
    def __init__(self, config, rnn_units=128):
        self.name = 'NerBiLSTM'
        self.config = config
        self.RNN_UNITS = rnn_units if rnn_units else config.RNN_UNITS
        self.create_model()
        
        
    def create_model(self):
        inputs = Input(shape=(self.config.MAXLEN, ), dtype='int32', name='inputs')  # (, MAXLEN)
        X = Embedding(self.config.VOCAB_SIZE, self.config.WORD_EMBED_DIM)(inputs)   # (, MAXLEN, WORD_EMBED_DIM)
        X = Bidirectional(LSTM(self.RNN_UNITS // 2, return_sequences=True))(X)      # (, MAXLEN, RNN_UNITS)
        X = Dropout(0.3)(X)
        out = CRF(self.config.N_TAGS, sparse_target=True, name='crf')(X)            # (, MAXLEN, N_TAGS)
        self.model = Model(inputs=inputs, outputs=out)
        self.model.summary()
