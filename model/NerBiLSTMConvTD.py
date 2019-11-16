# -*- coding: utf-8 -*-
"""
Created:    2019-11-16 14:14:05
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, LSTM, \
                Dropout, ZeroPadding1D, Conv1D, TimeDistributed, Dense, Concatenate
from keras.models import Model
from keras_contrib.layers import CRF


class NerBiLSTMConvTD:
    """
    Model: Random Embedding -> Concatenate(BiLSTM, Conv1D -> TD(Dense)) -> TD(Dense) -> CRF
    参考: <https://blog.csdn.net/xinfeng2005/article/details/78485748>
    """
    def __init__(self, config, rnn_units=128, cnn_units=64, pad_size=2, fc_units=32):
        self.config = config
        self.RNN_UNITS = rnn_units if rnn_units else config.RNN_UNITS
        self.PAD_SIZE = pad_size if pad_size else config.PAD_SIZE
        self.CNN_UNITS = cnn_units if cnn_units else config.CNN_UNITS
        self.FC_UNITS = fc_units if fc_units else config.FC_UNITS
        self.name = 'NerBiLSTMConvTD'
        
        
    def create_model(self):
        # 输入：Input -> Embedding
        inputs = Input(shape=(self.config.MAXLEN,), dtype='int32', name='word_input')   # (, MAXLEN)
        X = Embedding(self.config.VOCAB_SIZE, self.config.WORD_EMBED_DIM)(inputs)       # (, MAXLEN, WORD_EMBED_DIM)
        X = SpatialDropout1D(0.2)(X)
        
        # 分支1：Embedding -> BiLSTM -> Dropout => X1
        X1 = Bidirectional(LSTM(self.RNN_UNITS // 2, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(X)  # (, MAXLEN, RNN_UNITS)
        X1 = Dropout(0.1)(X1)
        
        # 分支2：Embedding -> ZeroPadding1D -> Conv1D -> Dropout -> TimeDistributed(Dense) => X2
        # 在Conv1D处理前，先用ZeroPadding手动Padding一下，以保证Conv1D后Seq维度不变  TODO 为何不直接在Conv1D中设置padding='same'？
        X2 = ZeroPadding1D(padding=self.PAD_SIZE)(X)                                         # (, MAXLEN + 2 * PAD_SIZE, WORD_EMBED_DIM)
        X2 = Conv1D(self.CNN_UNITS, kernel_size=2 * self.PAD_SIZE + 1, padding='valid')(X2)  # (, MAXLEN, CNN_UNITS)
        X2 = Dropout(0.1)(X2)
        X2 = TimeDistributed(Dense(self.FC_UNITS))(X2)                                       # (, MAXLEN, FC_UNITS)
        
        # 合并：Concatenate(X1, X2) -> TimeDistributed(Dense) -> CRF
        X = Concatenate()([X1, X2])                                         # (, MAXLEN, RNN_UNITS + FC_UNITS)
        X = TimeDistributed(Dense(self.config.N_TAGS))(X)                   # (, MAXLEN, N_TAGS)
        out = CRF(self.config.N_TAGS, sparse_target=True, name='crf')(X)    # (, MAXLEN, N_TAGS)
        
        model = Model(inputs=inputs, outputs=out)
        model.summary()
        return model