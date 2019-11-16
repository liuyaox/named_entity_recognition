# -*- coding: utf-8 -*-
"""
Created:    2019-11-16 13:36:14
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Input, Embedding, LSTM, Bidirectional, Dropout
from keras.models import Model
from keras_contrib.layers import CRF


class NerBiLSTM:
    """
    Model: Random Embedding -> BiLSTM -> CRF
    """
    def __init__(self, config, rnn_units=128):
        self.config = config
        self.RNN_UNITS = rnn_units if rnn_units else config.RNN_UNITS
        self.name = 'NerBiLSTM'
        
        
    def create_model(self):
        inputs = Input(shape=(self.config.MAXLEN, ), dtype='int32', name='inputs')  # (, MAXLEN)
        X = Embedding(self.config.VOCAB_SIZE, self.config.WORD_EMBED_DIM)(inputs)   # (, MAXLEN, WORD_EMBED_DIM)
        X = Bidirectional(LSTM(self.RNN_UNITS // 2, return_sequences=True))(X)      # (, MAXLEN, RNN_UNITS)
        X = Dropout(0.3)(X)
        out = CRF(self.config.N_TAGS, sparse_target=True, name='crf')(X)            # (, MAXLEN, N_TAGS)  # TODO sparse_target=True/False???
        model = Model(inputs=inputs, outputs=out)
        model.summary()
        return model