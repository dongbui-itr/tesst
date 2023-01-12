import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Layer
import math
import numpy as np


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(Layer):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(dropout)

        pe = np.zeros((max_len, d_model))
        position = tf.expand_dims(tf.range(0, max_len, dtype=tf.float32), axis=1)
        # div_term = tf.math.exp(tf.cast(tf.range(0, d_model, 2), dtype=tf.float32) * (-math.log(10000.0) / d_model))
        angle_rads = self.get_angles(position,
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pe = tf.transpose(tf.expand_dims(tf.convert_to_tensor(pe), axis=0), [1, 0, 2])
        # self.register_buffer('pe', pe)

    def call(self, x):
        x = x + tf.cast(tf.repeat(self.pe[:x.shape[0], :, :], repeats=x.shape[1], axis=1), dtype=tf.float32)
        return self.dropout(x)

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
