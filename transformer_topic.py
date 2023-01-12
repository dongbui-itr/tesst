import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

from transformer_layers import Encoder, Decoder, create_masks
from fnet import FNetEncoder

config = {"vocab_size": 32000, "hidden_size": 128, "embedding_size": 128, "intermediate_size": 2048,
          "max_position_embeddings": 512, "fourier": "fft", "pad_token_id": 3, "type_vocab_size": 4,
          "layer_norm_eps": 1e-12, "dropout_rate": 0.1, "num_hidden_layers": 4}


#########################################
#               MAIN MODEL              #
#########################################
class TopicTransformerModule(Layer):
    def __init__(self, d_model, nhead, num_layers, mlc, attention, vocab_size):
        super(TopicTransformerModule, self).__init__()

        # self.transformer_encoder = Encoder(num_layers=num_layers, d_model=d_model,
        #                                    num_heads=nhead, dff=2048, input_vocab_size=vocab_size,
        #                                    maximum_position_encoding=5000)
        self.transformer_encoder = FNetEncoder(config)

        self.mlc = mlc
        self.attention = attention

        self.transformer_decoder = Decoder(num_layers=num_layers, d_model=d_model * 2,
                                           num_heads=nhead, dff=2048, target_vocab_size=vocab_size,
                                           maximum_position_encoding=5000)

    def call(self, image_features, avg_feats, embed, tgt, training):
        attended_features = self.transformer_encoder(image_features, training=training)

        # attended : (batch, num_features, feature_size)
        def forward_attention(mlc, co_att, avg_features):
            tags, semantic_features = mlc(avg_features)
            ctx, alpht_v, alpht_a = co_att(avg_features, semantic_features)
            return tags, ctx

        tags, ctx = forward_attention(self.mlc, self.attention, avg_feats)

        contexts = tf.tile(tf.expand_dims(ctx, axis=1), [1, attended_features.shape[1], 1])
        attended_features = tf.concat([attended_features, contexts], axis=2)

        look_ahead_mask = create_masks(tgt)
        out, _ = self.transformer_decoder(embed, attended_features, training=training,
                                          look_ahead_mask=look_ahead_mask, padding_mask=None)
        return out, tags  # (seq, batch, embed)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = tf.transpose(tf.experimental.numpy.triu(tf.ones((sz, sz))), perm=[1, 0])
        mask = tf.where(mask == 0, tf.constant(-np.inf), 0)
        return mask
