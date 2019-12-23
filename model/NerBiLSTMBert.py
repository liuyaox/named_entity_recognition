#! -*- coding:utf-8 -*-
"""
Created:    2019-11-16 14:05:44
Author:     liuyao8
Descritipn: 
"""
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM
from keras.optimizers import Adam

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_bert import load_trained_model_from_checkpoint

from utils import pre, rec



class NerBiLSTMBert:
    """
    Model: BERT -> BiLSTM -> CRF
    参考：<https://github.com/UmasouTTT/keras_bert_ner/blob/master/Model/BertBilstmCrf.py>
    """
    def __init__(self, config, seq_maxlen=None, rnn_units=None, bert_path=None):
        self.name = 'NerBiLSTMBert'
        self.config = config
        self.seq_maxlen = seq_maxlen if seq_maxlen else config.SEQ_MAXLEN
        self.rnn_units = rnn_units if rnn_units else config.BERT_RNN_UNITS
        bert_path = bert_path if bert_path else config.bert_path
        self.build_model(bert_path)


    def build_model(self, bert_path):
        '''使用预训练BERT构建模型：BERT + BiLSTM + CRF'''
        bert = load_trained_model_from_checkpoint(
            bert_path + 'bert_config.json',
            bert_path + 'bert_model.ckpt',
            seq_len=self.seq_maxlen,
            trainable = True
        )
        bert.name = 'bert_model'
        X1 = Input(shape=(self.seq_maxlen, ), name='Input-Token')
        X2 = Input(shape=(self.seq_maxlen, ), name='Input-Segment')
        X = bert([X1, X2])
        X = Bidirectional(LSTM(self.rnn_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(X)
        out = CRF(self.config.N_TAGS, sparse_target=True, name='crf')(X)
        self.model = Model([X1, X2], out)
        self.model.summary()


    def set_bert_trainable(self, trainable=True):
        for layer in self.model.get_layer('bert_model').layers:
            layer.trainable = trainable

            
    def model_train(self, x_train, y_train, lr=1e-4, batch_size=32, epochs=10, train_plot=False, model_save=False):
        '''生成Train和Test，训练模型'''
        self.model.compile(optimizer=Adam(lr), loss=crf_loss, metrics=[crf_accuracy, pre, rec])
        history = self.model.fit(
                        x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.3
        )
        if train_plot:
            self.model_train_plot(history)
        if model_save:
            self.model.save(self.config.model_bert_file)
            
            
    def model_train_plot(self, history):
        '''训练过程可视化'''
        history_dict = history.history
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'yo', label='Training loss')
        plt.plot(epochs, val_loss, 'y', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend
        plt.show()

        
    def model_predict(self, sentence, dp, load_weights=False):
        '''模型应用：文本输入，预测出Tags'''
        if load_weights:
            self.model.load_weights(self.config.model_bert_file)
        sent_token_ids, sent_segment_ids = dp.encode_input_x([sentence])
        tag_probs = self.model.predict([sent_token_ids, sent_segment_ids])[0]
        tags = dp.decode_output(tag_probs[1: -1])   # 舍弃第1个和最后1个，分别对应[CLS]和[SEP]
        return tags



if __name__ == "__main__":

    pass