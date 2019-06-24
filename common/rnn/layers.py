import numpy as np

import common.util as util


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.cache = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.cache = x
        return out

    def backward(self, dout):
        W, b = self.params
        x = self.cache
        dx = np.dot(dout, W.T)
        dW = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        out = W[idx]
        self.idx = idx
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None


class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params

        u = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(u)

        self.cache = [x, h_prev, h_next]

        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        du = dh_next * (1 - h_next ** 2)

        dWh = np.dot(h_prev.T, du)
        dh_prev = np.dot(du, Wh.T)
        dWx = np.dot(x.T, du)
        dx = np.dot(du.Wx.T)
        db = np.sum(du, axis=0, keepdims=True)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = util.sigmoid(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = util.cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dl=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dl
        dx = dx / batch_size
        return dx