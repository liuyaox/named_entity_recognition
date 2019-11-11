# -*- coding: utf-8 -*-
"""
Created:    2019-11-10 20:43:17
Author:     liuyao8
Descritipn: 参考<https://blog.csdn.net/xinfeng2005/article/details/78485748>
"""

from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, LSTM, \
                Dropout, ZeroPadding1D, Conv1D, TimeDistributed, Dense, Concatenate
from keras.models import Model
from keras_contrib.layers import CRF


MAXLEN = 100
VOCAB_SIZE = 2500
EMBED_DIM = 128
HIDDEN_UNITS = 32
NUM_CLASS = 5


# 输入：Input -> Embedding
inputs = Input(shape=(MAXLEN,), dtype='int32', name='word_input')   # (, 100)
X = Embedding(VOCAB_SIZE, EMBED_DIM)(inputs)                        # (, 100, 128)
X = SpatialDropout1D(0.2)(X)

# 分支1：Embedding -> BiLSTM -> Dropout => X1
X1 = Bidirectional(LSTM(HIDDEN_UNITS, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(X)  # (, 100, 64)
X1 = Dropout(0.1)(X1)

# 分支2：Embedding -> ZeroPadding1D -> Conv1D -> Dropout -> TimeDistributed(Dense) => X2
HALF_WIN_SIZE = 2
X2 = ZeroPadding1D(padding=HALF_WIN_SIZE)(X)                            # (, 104, 128)  ???
X2 = Conv1D(50, kernel_size=2 * HALF_WIN_SIZE + 1, padding='valid')(X2) # (, 100, 50)
X2 = Dropout(0.1)(X2)
X2 = TimeDistributed(Dense(50))(X2)                                     # (, 100, 50)

# 合并：X1 + X2 -> Concatenate -> TimeDistributed(Dense) -> CRF
concat = Concatenate(axis=2)([X1, X2])              # (, 100, 114)
dense = TimeDistributed(Dense(NUM_CLASS))(concat)   # (, 100, 5)
crf = CRF(NUM_CLASS, sparse_target=False)           # (, 100, 5)
out = crf(dense)

model = Model(inputs=inputs, outputs=out)
model.summary()


# Compile & Train
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
checkpointer = ModelCheckpoint(filepath="bilstm_1102_k205_tf130.w", verbose=0, save_best_only=True, save_weights_only=True)
losshistory = LossHistory()
history = model.fit(x_train, y_train, batch_size=32, epochs=500, callbacks=[checkpointer, losshistory], verbose=1, validation_split=0.1)