# -*- coding: utf-8 -*-
"""
Created:    2019-11-10 20:20:30
Author:     liuyao8
Descritipn: 参考<http://www.voidcn.com/article/p-pykfinyn-bro.html>
            参考<https://github.com/UmasouTTT/keras_bert_ner/blob/master/Model/BertBilstmCrf.py>
"""
from numpy import expand_dims

from keras.layers import Input, Embedding, LSTM, Bidirectional, Dropout, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_contrib.utils import save_load_utils
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

MAXLEN = 100
VOCAB_SIZE = 2500
EMBED_DIM = 200
RNN_UNITS = 64
NUM_CLASS = 5
SEQ_MAXLEN = 80     # ?


# Embedding + BiLSTM + CRF
inputs = Input(shape=(MAXLEN, ), dtype='int32', name='inputs')  # (, 100)
X = Embedding(VOCAB_SIZE, EMBED_DIM)(inputs)                    # (, 100, 200)
X = Bidirectional(LSTM(RNN_UNITS, return_sequences=True))(X)    # (, 100, 128)
X = Dropout(0.3)(X)
X = Bidirectional(LSTM(RNN_UNITS, return_sequences=True))(X)    # (, 100, 128)
X = Dropout(0.3)(X)
X = TimeDistributed(Dense(NUM_CLASS))(X)                        # (, 100, 5)
crf_layer = CRF(NUM_CLASS)                                      # (, 100, 5)
out = crf_layer(X)
model = Model(inputs=inputs, outputs=out)
model.summary()


model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
model_path = 'xxx'
save_load_utils.save_all_weights(model, model_path)     # 保存模型
save_load_utils.load_all_weights(model, model_path)     # 加载模型



# BERT + BiLSTM + CRF
model_path = ".\\Parameter\\chinese_L-12_H-768_A-12\\"
bert = load_trained_model_from_checkpoint(
    model_path + "bert_config.json",
    model_path + "bert_model.ckpt",
    seq_len=SEQ_MAXLEN
)
for layer in bert.layers:
    layer.trainable = True
    
input1 = Input(shape=(None, ), name='word_labels_input')
input2 = Input(shape=(None, ), name='seq_types_input')
X = bert([input1, input2])
X = Bidirectional(LSTM(RNN_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(X)
out = CRF(NUM_CLASS, sparse_target=True)(X)
model = Model(inputs=[input1, input2], outputs=out)
model.summary()
model.compile(optimizer=Adam(1e-4), loss=crf_loss, metrics=[crf_accuracy])


#预处理输入X
def PreProcessInputData(text):
    tokenizer = Tokenizer(vocab)
    word_labels = []
    seq_types = []
    for sequence in text:
        code = tokenizer.encode(first=sequence, max_len=SEQ_MAXLEN)
        word_labels.append(code[0])
        seq_types.append(code[1])
    return word_labels, seq_types


#预处理输入Y
def PreProcessOutputData(text):
    tags = []
    for line in text:
        tag = [0]
        for item in line:
            tag.append(int(label[item.strip()]))
        tag.append(0)
        tags.append(tag)

    pad_tags = pad_sequences(tags, maxlen=SEQ_MAXLEN, padding="post", truncating="post")
    result_tags = expand_dims(pad_tags, 2)
    return result_tags