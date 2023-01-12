import tensorflow as tf


class FNetLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(FNetLayer, self).__init__()
        self.mixing_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config['layer_norm_eps'])
        self.feed_forward = tf.keras.layers.Dense(config['intermediate_size'])
        self.output_dense = tf.keras.layers.Dense(config['hidden_size'])
        self.output_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config['layer_norm_eps'])
        self.dropout = tf.keras.layers.Dropout(config['dropout_rate'])

    def call(self, hidden_states, training=False):
        fft_output = tf.math.real(tf.signal.fft(tf.signal.fft(tf.cast(hidden_states, tf.complex64))))
        fft_output = self.mixing_layer_norm(fft_output + hidden_states, training=training)
        intermediate_output = self.feed_forward(fft_output)
        intermediate_output = tf.keras.activations.gelu(intermediate_output)
        output = self.output_dense(intermediate_output)
        output = self.dropout(output, training=training)
        output = self.output_layer_norm(output + fft_output)

        return output


class FNetEncoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(FNetEncoder, self).__init__()
        self.config = config
        self.layers = [FNetLayer(config) for _ in range(config['num_hidden_layers'])]

    def call(self, hidden_states, training=False):
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, training)

        return hidden_states
