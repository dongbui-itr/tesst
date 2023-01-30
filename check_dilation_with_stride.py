import os
import numpy as np
from tensorflow import keras as k
import torch as t
import tensorflow as tf


def main(dilation=100, shape=1250):
    arr = np.arange(1, shape + 1, 1).reshape(1, 1, shape)
    # arr = np.ones((1, 10)).reshape(1, 1, 10)
    input = t.FloatTensor(arr)
    input_tf = tf.convert_to_tensor(arr, dtype=float)
    input_tf = tf.reshape(input_tf, (1, shape, 1))

    output_tf = k.layers.Conv1D(filters=1,
                                kernel_size=3,
                                strides=1, use_bias=False, kernel_initializer="ones",  # bias_initializer="ones",
                                dilation_rate=dilation,
                                padding="valid"
                                )(input_tf)

    index = np.arange(0, output_tf.shape[1] + 1)
    out_tf_config = output_tf[:, index[0]:index[-1]:2]

    conv1d_torch = t.nn.Conv1d(in_channels=1,
                               out_channels=1,
                               kernel_size=3,
                               stride=2,
                               dilation=dilation,
                               bias=False,
                               # padding="same"
                               )  # (input)
    conv1d_torch.weight.data.fill_(1)
    output = conv1d_torch(input)

    # t.nn.init.xavier_uniform(output.weight)

    # print(input)
    print("torch: ", output.shape)
    print("tf: ", output_tf.shape)
    print("tf change", out_tf_config.shape)
    a = 10


if __name__ == '__main__':
    main()
