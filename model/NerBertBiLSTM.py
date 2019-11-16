# -*- coding: utf-8 -*-
"""
Created:    2019-11-16 14:05:44
Author:     liuyao8
Descritipn: 
"""

from keras.layers import Input, Bidirectional, LSTM
from keras.models import Model
from keras_contrib.layers import CRF
from keras_bert import load_trained_model_from_checkpoint


class NerBertBiLSTM:
    """
    Model: BERT -> BiLSTM -> CRF
    参考: <https://github.com/UmasouTTT/keras_bert_ner/blob/master/Model/BertBilstmCrf.py>
    """
    def __init__(self, config, rnn_units=128):
        self.config = config
        self.RNN_UNITS = rnn_units if rnn_units else config.RNN_UNITS
        self.bert = load_trained_model_from_checkpoint(
            config.bert_model_path + "bert_config.json",
            config.bert_model_path + "bert_model.ckpt",
            seq_len=config.SEQ_MAXLEN
        )
        self.name = 'NerBertBiLSTM'
        
        
    def create_model(self):
        for layer in self.bert.layers:
            layer.trainable = True
        input1 = Input(shape=(None, ), name='word_labels_input')
        input2 = Input(shape=(None, ), name='seq_types_input')
        X = self.bert([input1, input2])
        X = Bidirectional(LSTM(self.RNN_UNITS, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(X)
        out = CRF(self.config.N_TAGS, sparse_target=True, name='crf')(X)
        model = Model(inputs=[input1, input2], outputs=out)
        model.summary()
        return model