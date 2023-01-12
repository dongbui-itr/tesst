import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Softmax, BatchNormalization, Activation, Layer


class MLC(Layer):
    def __init__(self,
                 classes=156,
                 sementic_features_dim=512,
                 k=10):
        super(MLC, self).__init__()
        self.classifier = Dense(classes)
        self.embed = Embedding(classes, sementic_features_dim)
        self.k = k
        self.softmax = Softmax(axis=-1)
        self.__init_weight()

    def __init_weight(self):
        self.classifier.kernel_initializer = tf.keras.initializers.RandomUniform(-0.1, 0.1)

    def call(self, avg_features):
        tags = self.softmax(self.classifier(avg_features))
        semantic_features = self.embed(tf.math.top_k(tags, self.k)[1])
        return tags, semantic_features


class CoAttention(Layer):
    def __init__(self,
                 embed_size=512,
                 hidden_size=512,
                 visual_size=2048,
                 k=10,
                 momentum=0.1):
        super(CoAttention, self).__init__()
        self.W_v = Dense(visual_size)
        self.bn_v = BatchNormalization(momentum=momentum)

        self.W_v_h = Dense(visual_size)
        self.bn_v_h = BatchNormalization(momentum=momentum)

        self.W_v_att = Dense(visual_size)
        self.bn_v_att = BatchNormalization(momentum=momentum)

        self.W_a = Dense(hidden_size)
        self.bn_a = BatchNormalization(momentum=momentum)

        self.W_a_h = Dense(hidden_size)
        self.bn_a_h = BatchNormalization(momentum=momentum)

        self.W_a_att = Dense(hidden_size)
        self.bn_a_att = BatchNormalization(momentum=momentum)

        # self.W_fc = nn.Linear(in_features=visual_size, out_features=embed_size)  # for v3
        self.W_fc = Dense(embed_size)
        self.bn_fc = BatchNormalization(momentum=momentum)

        self.tanh = Activation('tanh')
        self.softmax = Softmax(axis=-1)

        # self.__init_weight()

    def __init_weight(self):
        self.W_v.kernel_initializer = tf.keras.initializers.RandomUniform(-0.1, 0.1)

        self.W_v_h.kernel_initializer = tf.keras.initializers.RandomUniform(-0.1, 0.1)

        self.W_v_att.kernel_initializer = tf.keras.initializers.RandomUniform(-0.1, 0.1)

        self.W_a.kernel_initializer = tf.keras.initializers.RandomUniform(-0.1, 0.1)

        self.W_a_att.kernel_initializer = tf.keras.initializers.RandomUniform(-0.1, 0.1)

        self.W_fc.kernel_initializer = tf.keras.initializers.RandomUniform(-0.1, 0.1)

    def call(self, avg_features, semantic_features):
        W_v = self.bn_v(self.W_v(avg_features))

        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v))))
        v_att = tf.math.multiply(alpha_v, avg_features)

        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(W_a))))
        a_att = tf.math.reduce_sum(tf.math.multiply(alpha_a, semantic_features), axis=1)

        # ctx = self.W_fc(tf.keras.layers.concatenate([v_att, a_att], axis=1))
        ctx = self.W_fc(tf.concat([v_att, a_att], 1))

        return ctx, alpha_v, alpha_a
