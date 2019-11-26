# -*- coding: utf-8 -*-
"""
Created:    2019-11-16 13:56:09
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, TimeDistributed, Dense
from keras.models import Model
from keras_contrib.layers import CRF


class NerBiLSTMsTD:
    """
    Model: Random Embedding -> BiLSTMs -> TD(Dense) -> CRF
    参考: <http://www.voidcn.com/article/p-pykfinyn-bro.html>
    """
    def __init__(self, config, rnn_units=128):
        self.name = 'NerBiLSTMsTD'
        self.config = config
        self.RNN_UNITS = rnn_units if rnn_units else config.RNN_UNITS
        self.create_model()
        
        
    def create_model(self):
        inputs = Input(shape=(self.config.MAXLEN, ), dtype='int32', name='inputs')  # (, MAXLEN)
        X = Embedding(self.config.VOCAB_SIZE, self.config.WORD_EMBED_DIM, mask_zero=True)(inputs)   # (, MAXLEN, WORD_EMBED_DIM)
        X = Bidirectional(LSTM(self.RNN_UNITS // 2, return_sequences=True))(X)      # (, MAXLEN, RNN_UNITS)
        X = Dropout(0.3)(X)
        X = Bidirectional(LSTM(self.RNN_UNITS // 2, return_sequences=True))(X)      # (, MAXLEN, RNN_UNITS)
        X = Dropout(0.3)(X)
        X = TimeDistributed(Dense(self.config.N_TAGS))(X)                           # (, MAXLEN, N_TAGS)
        out = CRF(self.config.N_TAGS, sparse_target=True, name='crf')(X)            # (, MAXLEN, N_TAGS)
        self.model = Model(inputs=inputs, outputs=out)
        self.model.summary()
