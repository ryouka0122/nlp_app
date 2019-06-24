# -*- coding: utf-8 -*-
import numpy as np
from common.rnn.time_layers import TimeAffine, TimeRNN, TimeEmbedding, TimeSoftmaxWithLoss


class SimpleRnn:
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size):
        v, e, h, o = vocab_size, embedding_size, hidden_size, output_size
        randn = np.random.randn

        embed_w = randn(v, e).astype('f')
        rnn_wh = randn(h, h).astype('f')
        rnn_wx = randn(e, h).astype('f')
        rnn_b = randn(h).astype('f')
        affine_w = randn(h, o).astype('f')
        affine_b = randn(o).astype('f')

        self.rnn_layer = TimeRNN(rnn_wh, rnn_wx, rnn_b, stateful=True)
        self.layers = [
            TimeEmbedding(embed_w),
            self.rnn_layer,
            TimeAffine(affine_w, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()
