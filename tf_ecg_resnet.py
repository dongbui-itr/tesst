import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models


class BasicBlock(layers.Layer):
    """
    This class implements a residual block.
    """

    def __init__(self, in_channels, out_channels, stride, dropout, dilation, **kwargs):
        """
        Initializes BasicBlock object.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            stride (int): stride of the convolution.
            dropout (float): probability of an argument to get zeroed in the
                dropout layer.
            dilation (int): amount of dilation in the dilated branch.
        """
        super(BasicBlock, self).__init__(**kwargs)
        self.kernel_size = 5
        self.num_branches = 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dropout = dropout
        self.dilation = dilation

        # branch 0
        self.branch0_batch1 = layers.BatchNormalization()
        self.branch0_conv1d1 = layers.SeparableConv1D(self.in_channels // self.num_branches, kernel_size=1, padding='valid',
                                             strides=1,
                                             use_bias=False)
        self.branch0_batch2 = layers.BatchNormalization()
        self.branch0_padding_1 = layers.ZeroPadding1D(padding=(self.kernel_size - 1) // 2)
        self.branch0_conv1d2 = layers.SeparableConv1D(self.out_channels // self.num_branches, kernel_size=self.kernel_size,
                                             padding='valid',
                                             strides=self.stride, use_bias=False)

        # branch 1
        self.branch1_batch1 = layers.BatchNormalization()
        self.branch1_conv1d1 = layers.SeparableConv1D(self.in_channels // self.num_branches, kernel_size=1, padding='valid',
                                             strides=1, dilation_rate=1,
                                             use_bias=False)
        self.branch1_batch2 = layers.BatchNormalization()
        self.branch1_padding_1 = layers.ZeroPadding1D(padding=((self.kernel_size - 1) * dilation) // 2 - 25)
        self.branch1_padding_1_1 = layers.ZeroPadding1D(padding=((self.kernel_size - 1) * dilation) // 2)
        self.branch1_padding_1_2 = layers.ZeroPadding1D(padding=((self.kernel_size - 1) * (dilation) // 3 + 1) + 104)
        self.branch1_conv1d2 = layers.SeparableConv1D(self.out_channels // self.num_branches, kernel_size=self.kernel_size,
                                             padding='valid',
                                             strides=self.stride, dilation_rate=self.dilation, use_bias=False)
        # self.branch1_batch3 = layers.BatchNormalization()
        # self.branch1_padding_2 = layers.ZeroPadding1D(padding=2)
        # self.branch1_conv1d3 = layers.Conv1D(self.out_channels // self.num_branches, kernel_size=self.kernel_size,
        #                                      padding='valid',
        #                                      strides=self.stride, use_bias=False)

        # shortcut
        self.shortcut_conv1d = layers.SeparableConv1D(self.out_channels, kernel_size=1, padding='valid', strides=self.stride,
                                             use_bias=False)

    def sub_branch0(self, x, training=False):
        x = self.branch0_batch1(x, training=training)
        x = layers.Activation('relu')(x)
        x = self.branch0_conv1d1(x)
        x = self.branch0_batch2(x, training=training)
        x = layers.Activation('relu')(x)
        if training:
            x = layers.Dropout(self.dropout)(x)
        x = self.branch0_padding_1(x)
        x = self.branch0_conv1d2(x)
        return x

    def sub_branch1(self, x, training=False):
        x = self.branch1_batch1(x, training=training)
        x = layers.Activation('relu')(x)
        x = self.branch1_conv1d1(x)
        x = self.branch1_batch2(x, training=training)
        x = layers.Activation('relu')(x)
        if training:
            x = layers.Dropout(self.dropout)(x)
        x = self.branch1_padding_1(x)
        #print(x.shape)
        x = self.branch1_conv1d2(x)
        print(x)
        # x = self.branch1_batch3(x, training=training)
        # x = layers.Activation('relu')(x)
        # x = self.branch1_padding_2(x)
        # x = self.branch1_conv1d3(x)
        return x

    def sub_branch11(self, x, training=False):
        x = self.branch1_batch1(x, training=training)
        x = layers.Activation('relu')(x)
        x = self.branch1_conv1d1(x)
        x = self.branch1_batch2(x, training=training)
        x = layers.Activation('relu')(x)
        if training:
            x = layers.Dropout(self.dropout)(x)
        x = self.branch1_padding_1_1(x)
        #print(x.shape)
        x = self.branch1_conv1d2(x)
        print('Branch1_1 {}'.format(x.shape))
        return x

    def sub_branch12(self, x, training=False):
        x = self.branch1_batch1(x, training=training)
        x = layers.Activation('relu')(x)
        x = self.branch1_conv1d1(x)
        x = self.branch1_batch2(x, training=training)
        x = layers.Activation('relu')(x)
        if training:
            x = layers.Dropout(self.dropout)(x)
        x = self.branch1_padding_1_2(x)
        #print(x.shape)
        x = self.branch1_conv1d2(x)
        print('Branch1_2 {}'.format(x.shape))
        return x

    def shortcut(self, x):
        if self.in_channels == self.out_channels and self.stride == 1:
            shortcut = x
        else:
            shortcut = self.shortcut_conv1d(x)
        return shortcut

    def call(self, x, training=False):
        x0 = self.sub_branch0(x, training=training)
        x1 = self.sub_branch1(x, training=training)
        #shortcut = self.shortcut(x)
        x11 = self.sub_branch11(x, training=training)
        x12 = self.sub_branch12(x, training=training)

        print(x0.shape, x1.shape, x11.shape, x12.shape)
        try:
            x = tf.concat([x0, x1], 2)
        except:
            try:
                x = tf.concat([x0, x11], 2)
            except:
                x = tf.concat([x0, x12], 2)
        x = tf.keras.layers.Add()([x, x])

        return x


class Stem(layers.Layer):
    def __init__(self, first_width, dropout):
        """
        Initializes Stem object.

        Args:
            first_width (int): the output width of the stem.
            dropout (float): the dropout probability.
        """
        super(Stem, self).__init__(name='Stem')
        self.first_width = first_width
        self.dropout = dropout

        self.padding_1 = layers.ZeroPadding1D(padding=3)
        self.conv1d1 = layers.SeparableConv1D(self.first_width // 2, kernel_size=7, padding='valid', strides=2, dilation_rate=1,
                                     use_bias=False)
        self.batch1 = layers.BatchNormalization()
        self.conv1d2 = layers.SeparableConv1D(self.first_width, kernel_size=1, padding='valid', strides=1, use_bias=False)
        self.batch2 = layers.BatchNormalization()
        self.padding_2 = layers.ZeroPadding1D(padding=2)
        self.conv1d3 = layers.SeparableConv1D(self.first_width, kernel_size=5, padding='valid', strides=1, use_bias=False)

    def call(self, inputs, training=False):
        x = self.padding_1(inputs)
        x = self.conv1d1(x)
        x = self.batch1(x, training=training)
        x = layers.Activation('relu')(x)
        x = self.conv1d2(x)
        x = self.batch2(x, training=training)
        x = layers.Activation('relu')(x)
        if training:
            x = layers.Dropout(self.dropout)(x)
        x = self.padding_2(x)
        x = self.conv1d3(x)

        return x


class ECGResNet(keras.Model):
    """
   This class implements the ECG-ResNet in Tensorflow.
   It handles the different layers and parameters of the model.
   Once initialized an ResNet object can perform forward.
   """

    def __init__(self, in_length, in_channels, n_grps, N, dropout,
                 first_width, stride, dilation):
        """
        Initializes ECGResNet object.

        Args:
            in_length (int): the length of the ECG signal input.
            in_channels (int): number of channels of input (= leads).
            n_grps (int): number of ResNet groups.
            N (int): number of blocks per groups.
            stride (tuple): tuple with stride value per block per group.
            dropout (float): the dropout probability.
            first_width (int): the output width of the stem.
            dilation (int): the space between the dilated convolutions.
        """
        super(ECGResNet, self).__init__(name='ECGResNet')
        self.in_length = in_length
        self.in_channels = in_channels
        self.n_grps = n_grps
        self.N = N
        self.stride = stride
        self.dropout = dropout
        self.first_width = first_width
        self.dilation = dilation

        self.num_branches = 2
        self.first_width = self.first_width * self.num_branches

        self.stem = Stem(self.first_width, self.dropout)

        self.basic_block_layers = []
        widths = [self.first_width]
        for grp in range(self.n_grps):
            widths.append(self.first_width * 2 ** grp)
        for grp in range(self.n_grps):
            in_channels = widths[grp]
            out_channels = widths[grp + 1]
            for i in range(self.N):
                self.basic_block_layers.append(BasicBlock(in_channels=(in_channels if i == 0 else out_channels),
                                                          out_channels=out_channels,
                                                          stride=self.stride[i],
                                                          dropout=self.dropout,
                                                          dilation=self.dilation))
        self.basic_block_layers.append(layers.BatchNormalization())
        self.basic_block_layers.append(layers.Activation('relu'))

    def make_group(self, x, training=False):
        for layer in self.basic_block_layers:
            x = layer(x, training=training)

        return x

    def call(self, x, training=False):
        """
        Performs forward pass of the input. Here an input tensor x is
        transformed through several layer transformations.

        Args:
            x (tensor): input to the block with size NxLxC
        Returns:
            out (tuple): outputs of forward pass, first the auxilliary halfway,
                then the final prediction
        """
        x = self.stem(x, training=training)
        x = self.make_group(x, training=training)
        kernel = layers.AveragePooling1D(pool_size=x.shape[1])
        average_feature = tf.squeeze(kernel(x))

        return x, average_feature

    def summary(self, input_shape):
        x = layers.Input(shape=input_shape)
        model = Model(inputs=x, outputs=[self.call(x)])
        return model.summary()


