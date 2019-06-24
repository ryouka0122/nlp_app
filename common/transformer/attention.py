# -*- coding: utf-8 -*-
import tensorflow as tf


class SimpleAttention(tf.keras.models.Model):
    def __init(self, depth: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth

        self.q_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='v_dense_layer')

        self.output_dense_layer = tf.keras.layers.Dense(depth, use_bias=False, name='output_dense_layer')

    def call(self, input: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        q = self.q_dense_layer(input)
        k = self.k_dense_layer(memory)
        v = self.v_dense_layer(memory)

        logit = tf.matmul(q, k, transpose_b=True)
        attention_weight = tf.nn.softmax(logit, name='attention_weight')

        attention_output = tf.matmul(attention_weight, v)
        return self.output_dense_layer(attention_output)


class MultiHeadAttention(tf.keras.models.Model):
    def __init__(self, hidden_dim: int, head_num: int, dropout_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        self.q_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='q_dense_layer')
        self.k_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='k_dense_layer')
        self.v_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='v_dense_layer')

        self.output_dense_layer = tf.keras.layers.Dense(hidden_dim, use_bias=False, name='output_dense_layer')
        self.attention_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(
            self,
            t_input: tf.Tensor,
            memory: tf.Tensor,
            attention_mask: tf.Tensor,
            training: bool
    ) -> tf.Tensor:
        q = self.q_dense_layer(t_input)
        k = self.k_dense_layer(memory)
        v = self.v_dense_layer(memory)

        # -- -- Multi-head Attention(split) -- -- #
        q = self._split_head(q)
        k = self._split_head(k)
        v = self._split_head(v)

        depth = self.hidden_dim // self.head_num

        # -- -- Scaled Dot Production -- -- #
        q *= depth ** -0.5  # for scaled dot production

        logit = tf.matmul(q, k, transpose_b=True)

        # -- -- Mask -- -- #
        logit += tf.to_float(attention_mask) * t_input.dtype.min  # mask (input.dtype.min is like -INF)

        attention_weight = tf.nn.softmax(logit, name='attention_weight')

        # -- -- Dropout -- -- #
        attention_weight = self.attention_dropout_layer(attention_weight, training=training)

        attention_output = tf.matmul(attention_weight, v)

        # -- -- Multi-head Attention(combine) -- -- #
        attention_output = self._combine_head(attention_output)

        return self.output_dense_layer(attention_output)

    def _split_head(self, x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('split_head'):
            batch_size, length, hidden_dim = tf.unstack(tf.shape(x))
            x = tf.reshape(x, [
                batch_size,
                length,
                self.head_num,
                self.hidden_dim // self.head_num
            ])
            return tf.transpose(x, [0, 2, 1, 3])

    def _combine_head(self, x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('combine_head'):
            batch_size, _, length, _ = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, batch_size, length, self.hidden_dim)


class SelfAttention(MultiHeadAttention):
    def call(
            self,
            t_input: tf.Tensor,
            attention_mask: tf.Tensor,
            training: bool
    ) -> tf.Tensor:
        return super().call(
            t_input=t_input,
            memory=t_input,
            attention_mask=attention_mask,
            training=training
        )




