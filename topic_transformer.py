import copy
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.numpy_ops import np_config

from ecg_resnet import ECGResNet
from network_topic import MLC, CoAttention
from transformer_topic import TopicTransformerModule
import numpy as np
from eval import evaluate_for_confusion
import pickle
import csv
from utils_model import get_next_word

np_config.enable_numpy_behavior()


class TopicTransformer(keras.Model):
    def __init__(self, vocab, in_length, in_channels,
                 n_grps, N, num_classes, k,
                 dropout, first_width,
                 stride, dilation, dilation_rate, num_layers, d_mode, nhead, **kwargs):
        super(TopicTransformer, self).__init__()
        self.vocab = vocab

        self.ecg_resnet_model = ECGResNet(in_length, in_channels,
                                          n_grps, N,
                                          dropout, first_width,
                                          stride, dilation, dilation_rate)


        self.feature_embedding = layers.Dense(d_mode)
        self.embed = layers.Embedding(len(vocab), 2 * d_mode)

        mlc = MLC(classes=num_classes, sementic_features_dim=d_mode, k=k)
        attention = CoAttention(embed_size=d_mode, hidden_size=d_mode, visual_size=256, k=k)

        self.transformer = TopicTransformerModule(d_mode, nhead, num_layers, mlc, attention, vocab_size=len(vocab))

        self.to_vocab = layers.Dense(len(vocab))

        self.mse_criterion = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self.train_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

        self.val_confusion_matrix = np.zeros((4, 10))
        self.list_val_F1_score = []

    def sample(self, waveforms, specs, sample_method, max_length):
        nb_batch = waveforms.shape[0]
        waveforms = tf.cast(waveforms, tf.float32)

        image_features, avg_features = self.ecg_resnet_model(waveforms)

        image_features = self.feature_embedding(image_features)

        start_tokens = np.array([self.vocab('<start>')])
        start_tokens = np.tile(start_tokens, (nb_batch, 1))
        start_tokens = tf.convert_to_tensor(start_tokens)
        sent = self.embed(start_tokens)

        tgt_mask = tf.zeros((sent.shape[0], 1), dtype=bool)
        y_out = tf.zeros([nb_batch, max_length]).numpy()

        for i in range(max_length):
            out, _ = self.transformer(image_features, avg_features, sent, tf.cast(sent[:, :, i], dtype=tf.int64),
                                      training=False)
            out = self.to_vocab(out[:, i, :])
            s = sample_method
            word_idx, props = get_next_word(out, temp=s['temp'], k=s['k'], p=s['p'], greedy=s['greedy'], m=s['m'])
            y_out[:, i] = word_idx.numpy()[:, 0]

            if i < max_length - 1:
                tgt_mask = tgt_mask | (word_idx == self.vocab('<end>'))

                embedded = self.embed(word_idx)
                sent = tf.concat([sent, embedded], 1)
                if tf.reduce_sum(tf.cast(tgt_mask, tf.int32)) == nb_batch:
                    break
            else:
                break

        return tf.convert_to_tensor(y_out)

    def call(self, inputs, training=False):
        waveforms, specs, targets = inputs
        image_features, avg_feats = self.ecg_resnet_model(waveforms, training)
        image_features = self.feature_embedding(image_features)
        embedded = self.embed(targets)
        out, tags = self.transformer(image_features, avg_feats, embedded, targets, training)
        vocab_distribution = self.to_vocab(out)

        return vocab_distribution, tags

    def get_config(self):
        string_of_bytes_vocab = pickle.dumps(self.vocab, 0)
        return {'vocab': str(string_of_bytes_vocab)}

    def loss_tags(self, label, tags):
        tag_loss = self.mse_criterion(label, tags)
        return tag_loss

    def xloss(self, out, targets, weights, tags, topic, type_weight=None, ignore_index=-1):
        out = tf.nn.log_softmax(out).reshape(-1, len(self.vocab))
        target = targets[:, 1:]
        batch_size, seq_length = target.shape
        target = tf.reshape(target, shape=(-1))

        loss = self.nllloss(target, out)
        loss = tf.reshape(loss, shape=(batch_size, seq_length))
        if type_weight:
            if type_weight == 's':
                alpha = tf.reshape(weights, shape=(batch_size, 1))
            else:
                alpha = copy.deepcopy(tf.convert_to_tensor(self.vocab.weights)[target])

            pt = tf.exp(-loss)
            gamma = 2
            loss = alpha * (1 - pt) ** gamma * loss

        loss_with_ignore_index = []
        for i in range(batch_size):
            loss_with_ignore_index.append(tf.reduce_sum(loss[i][targets[i, 1:] != ignore_index]))
        loss_with_ignore_index = tf.convert_to_tensor(loss_with_ignore_index)

        target_loss = self.loss_tags(topic, tags)

        return target_loss * 0.3 + loss_with_ignore_index * 0.7

    def check_prediction(self, out, targets):
        nb_batch = out.shape[0]
        index = random.randint(0, nb_batch - 1)

        idx = np.argmax(out[index], axis=1)
        pred = ' '.join([self.vocab.idx2word[idxn] for idxn in idx])
        truth = ' '.join([self.vocab.idx2word[word] for word in targets.numpy()[index]])

        print(f'\nTrue: {truth}')
        print(f'Pred: {pred}\n')

    def train_step(self, data):
        waveforms, specs, ids, targets, weights, labels = data[0]

        with tf.GradientTape() as tape:
            vocab_distribution, tags = self([waveforms, specs, targets], training=True)
            vocab_distribution = vocab_distribution[:, :-1, :]
            # Compute loss
            loss = self.xloss(vocab_distribution, targets, weights, tags, labels, type_weight=None,
                              ignore_index=self.vocab.word2idx['<pad>'])

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Check gradients
        self.check_gradient(gradients, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Check prediction
        # self.check_prediction(vocab_distribution, targets)

        self.train_loss_tracker.update_state(loss)

        return {'loss': self.train_loss_tracker.result()}

    def test_step(self, data):
        waveforms, specs, ids, targets, weights, labels = data[0]
        vocab_distribution, tags = self([waveforms, specs, targets])
        vocab_distribution = vocab_distribution[:, :-1, :]
        # Compute loss
        loss = self.xloss(vocab_distribution, targets, weights, tags, labels, type_weight=None,
                          ignore_index=self.vocab.word2idx['<pad>'])

        sample_method = {'temp': None, 'k': None, 'p': None, 'greedy': True, 'm': None}
        max_length = 50
        words = self.sample(waveforms, specs, sample_method, max_length)
        generated = self.vocab.decode(words, skip_first=False)
        truth = self.vocab.decode(targets)
        gts = {}
        res = {}
        ids = ids.numpy()
        for i in range(waveforms.shape[0]):
            res[ids[i]] = [generated[i]]
            gts[ids[i]] = [truth[i]]
            sub_confusion_matrix, check_predict = evaluate_for_confusion(gts[ids[i]][0], res[ids[i]][0], ignore='sinus')
            self.val_confusion_matrix += sub_confusion_matrix

        self.val_loss_tracker.update_state(loss)

        return {'loss': self.val_loss_tracker.result()}

    @staticmethod
    def check_gradient(gradients, trainable_vars):
        count = 0
        layers_without_gradient = []
        for (grad, var) in zip(gradients, trainable_vars):
            if grad is not None:
                count += 1
            else:
                layers_without_gradient.append(var.name)
        if count != len(gradients):
            print(f'Total layers {len(gradients)} - Layers with gradient {count}')
            raise Exception('No gradient at layers:', layers_without_gradient)

    @staticmethod
    def nllloss(labels, log_probs):
        """ Negative log likelihood."""
        return -(log_probs[range(len(labels)), labels])

    # @staticmethod
    # def nllloss(target, out):
    #     target_one_hot = tf.one_hot(target, out.shape[-1]) == 1
    #     return -(out[target_one_hot])


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open('./results/tf/log_val_cf.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(np.array(['Epochs', epoch + 1]))
            event_types = np.array(
                ['', 'AFIB', 'AVB2', 'AVB3', 'SINUS', 'SVT', 'VT', 'TACHY', 'BRADY', 'PAUSE', 'OTHER'])
            writer.writerow(event_types)
            for i in range(4):
                if i == 0:
                    row = np.insert(self.model.val_confusion_matrix[i].astype(str), 0, 'TP')
                elif i == 1:
                    row = np.insert(self.model.val_confusion_matrix[i].astype(str), 0, 'FN')
                elif i == 2:
                    row = np.insert(self.model.val_confusion_matrix[i].astype(str), 0, 'FP')
                else:
                    row = np.insert(self.model.val_confusion_matrix[i].astype(str), 0, 'TN')
                writer.writerow(row)
            SE = np.nan_to_num(
                self.model.val_confusion_matrix[0] / (
                        self.model.val_confusion_matrix[0] + self.model.val_confusion_matrix[1]))
            P = np.nan_to_num(
                self.model.val_confusion_matrix[0] / (
                        self.model.val_confusion_matrix[0] + self.model.val_confusion_matrix[2]))
            F1 = np.nan_to_num(2 * SE * P / (SE + P))
            SE = np.append(SE, np.mean(SE))
            P = np.append(P, np.mean(P))
            F1 = np.append(F1, np.mean(F1))
            self.model.list_val_F1_score.append(np.mean(F1))
            writer.writerow(np.insert(SE.astype(str), 0, 'SE'))
            writer.writerow(np.insert(P.astype(str), 0, 'P+'))
            writer.writerow(np.insert(F1.astype(str), 0, 'F1'))

            writer.writerow(np.array(['']))
        f.close()

        self.model.val_confusion_matrix = np.zeros((4, 10))

    def on_train_begin(self, logs=None):
        if not os.path.exists(os.path.join(os.getcwd(), 'results')):
            os.mkdir(os.path.join(os.getcwd(), 'results'))
        with open('./results/tf/log_val_cf.csv', 'w') as f:
            pass
        f.close()

    def on_train_end(self, logs=None):
        max_F1 = max(self.model.list_val_F1_score)
        argmax_f1 = self.model.list_val_F1_score.index(max_F1)
        with open('./results/tf/log_val_cf.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(np.array(
                ['', '', '', '', '', '', '', '', '', '', '', '', '',
                 f'Best model at epoch {argmax_f1 + 1} with F1 score {max_F1}']))
        f.close()
        print(f'Best model at epoch {argmax_f1 + 1} with F1 score {max_F1}')
