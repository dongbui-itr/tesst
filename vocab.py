import numpy as np
import tensorflow as tf


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.weights = tf.convert_to_tensor([])
        self.idx = 0

    def add_word(self, word, weight):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.weights = tf.concat([self.weights, tf.convert_to_tensor([weight])], axis=0)
            self.idx += 1

    def get_word(self, idx):
        return self.idx2word[idx]

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def decode(self, word_idxs, listfy=False, join_words=True, skip_first=True):
        captions = []
        for wis in word_idxs:
            caption = []
            if skip_first:
                wis = wis[1:]
            for wi in wis:
                word = self.idx2word[int(wi)]
                if word == '<end>':
                    break
                caption.append(word)
            if join_words:
                caption = ' '.join(caption)
            if listfy:
                caption = [caption]
            captions.append(caption)
        return captions
