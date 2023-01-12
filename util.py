import copy
import os

from nltk.tokenize import RegexpTokenizer
from collections import Counter

import tensorflow as tf
import numpy as np
import pandas as pd
import json

from dataset import collate_fn
from vocab import Vocabulary
import wfdb
import librosa

from eval import create_phrase_index


class RealDataset(tf.keras.utils.Sequence):
    def __init__(self, length, topic, vocab, train, waveform_dir, in_length, num_classes, dataset, batch_size,
                 label='Label'):
        self.topic = topic
        self.dataset = dataset
        self.waveform_dir = waveform_dir
        self.in_length = in_length
        self.num_classes = num_classes
        self.length = length
        self.batch_size = batch_size
        self.label = label
        self.tokenizer = RegexpTokenizer('\d+\.?,?\d+|-?/?\w+-?/?\w*|\w+|\d+|<[A-Z]+>')
        self.weights = self.setup_weights(self.dataset['Label'])
        if train:
            self.vocab = self.setup_vocab(self.dataset['Label'])
        else:
            self.vocab = vocab

        self.indexes = np.arange(self.length)

    def setup_vocab(self, labels):
        corpus = labels.str.cat(sep=" ")

        counter = Counter(self.tokenizer.tokenize(corpus))
        del counter['']

        abnormal_groups = ['atrial', 'fibrillation', 'flutter', 'supraventricular', 'tachycardia', 'paroxysmal',
                           'ventricular', 'run', '1st', 'degree', '2nd', '3rd', 'advanced', 'heart', 'block',
                           'high', 'grade', 'ivcd', 'pause', 'urgent', 'emergent', 'svt', 'psvt',
                           'fibrillation/flutter', 'av']

        normal_groups = ['artifact', 'sinus', 'arrhythmia', 'bradycardia', 'rhythm']

        counter = counter.most_common()
        words = []
        cnts = []
        for i in range(len(counter)):
            words.append(counter[i][0])
            if words[-1] in abnormal_groups:
                cnts.append(0.9)
            elif words[-1] in normal_groups:
                cnts.append(0.25)
            else:
                cnts.append(0.1)

        vocab = Vocabulary()
        vocab.add_word('<pad>', min(cnts))
        vocab.add_word('<start>', min(cnts))
        vocab.add_word('<end>', min(cnts))
        vocab.add_word('<unk>', min(cnts))

        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            vocab.add_word(word, float(cnts[i]))
        return vocab

    def setup_weights(self, labels):
        weights = np.zeros(len(labels))
        for i in range(0, len(labels)):
            if 'fibrillation' in labels[i] or 'vt' in labels[i] or '1st' in labels[i] or '2nd' in labels[i] \
                    or '3rd' in labels[i] or 'pause' in labels[i]:
                weights[i] = 0.75
            elif 'sinus' in labels[i] or 'artifact' in labels[i]:
                weights[i] = 0.25
            else:
                raise ValueError(f'Error at labels {labels[i]} {i}')

        return weights

    def __len__(self):
        return self.length // self.batch_size

    def __getitem__(self, index):
        if index == 0:
            np.random.shuffle(self.indexes)
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return next(self.data_generation(indexes))

    def data_generation(self, indexes):
        samples = []
        for idx in indexes:
            waveform, spec, sample_id = self.get_waveform(idx)

            sample = {
                'waveform': waveform,
                'spec': spec,
                'id': sample_id,
                'weights': self.weights[idx]
            }

            if self.label in self.dataset.columns.values:
                sentence = self.dataset[self.label].iloc[idx]
                try:
                    tokens = self.tokenizer.tokenize(sentence)
                except:
                    print(sentence)
                    raise Exception()
                vocab = self.vocab
                caption = [vocab('<start>')]
                caption.extend([vocab(token) for token in tokens])
                caption.append(vocab('<end>'))
                target = tf.convert_to_tensor(caption)

                sample['label'] = target

            if self.topic:
                sample['extra_label'] = tf.convert_to_tensor(create_phrase_index(self.dataset['Label'][idx]),
                                                             dtype=tf.float32)

            samples.append(sample)
        yield collate_fn(samples)

    def get_waveform(self, idx):
        try:
            raw_signal, fields = wfdb.rdsamp(os.path.join(self.waveform_dir, self.dataset['Path'][idx])
                                             + '/' + self.dataset['EventID'][idx])
        except:
            raw_signal, fields = wfdb.rdsamp(self.waveform_dir + '/' + self.dataset['EventID'][idx])

        channel = self.dataset['Channel'][idx]
        waveform = np.array(raw_signal[:, channel])

        if os.path.exists(os.path.join(self.waveform_dir, self.dataset['EventID'][idx]) + '.rev'):
            with open(os.path.join(self.waveform_dir, self.dataset['EventID'][idx]) + '.rev') as f:
                s = f.read()
        else:
            with open(os.path.join(self.waveform_dir, self.dataset['EventID'][idx]) + '.rhy') as f:
                s = f.read()

        ann = json.loads(s)
        if not len(ann['SampleMark']) and not len(ann['SampleMark2']):
            mark = np.random.choice(range(5000), 1, replace=False)[0]
            waveform = waveform[mark:mark + 5000]
        elif len(ann['SampleMark']) and not len(ann['SampleMark2']):
            len_sample_mark = ann['SampleMark'][0]['endSample'] - ann['SampleMark'][0]['startSample'] + 1
            start_sample = ann['SampleMark'][0]['startSample']
            end_sample = ann['SampleMark'][0]['endSample']
            if len_sample_mark != 5000:
                waveform = extend_array(waveform, start_sample, end_sample, 5000)
            else:
                waveform = waveform[start_sample:end_sample + 1]
        elif not len(ann['SampleMark']) and len(ann['SampleMark2']):
            len_sample_mark = ann['SampleMark2'][0]['endSample'] - ann['SampleMark2'][0]['startSample'] + 1
            start_sample = ann['SampleMark2'][0]['startSample']
            end_sample = ann['SampleMark2'][0]['endSample']
            if len_sample_mark != 5000:
                waveform = extend_array(waveform, start_sample, end_sample, 5000)
            else:
                waveform = waveform[start_sample:end_sample + 1]
        else:
            len_sample_mark = ann['SampleMark'][0]['endSample'] - ann['SampleMark'][0]['startSample'] + 1
            len_sample_mark2 = ann['SampleMark2'][0]['endSample'] - ann['SampleMark2'][0]['startSample'] + 1
            waveform_1 = waveform[ann['SampleMark'][0]['startSample']:ann['SampleMark'][0]['endSample'] + 1]
            waveform_2 = waveform[ann['SampleMark2'][0]['startSample']:ann['SampleMark2'][0]['endSample'] + 1]
            if len_sample_mark < 2500:
                waveform_1 = extend_array(waveform, ann['SampleMark'][0]['startSample'],
                                          ann['SampleMark'][0]['endSample'], 2500)
            if len_sample_mark2 < 2500:
                waveform_2 = extend_array(waveform, ann['SampleMark2'][0]['startSample'],
                                          ann['SampleMark2'][0]['endSample'], 2500)
            if 2500 < len_sample_mark < 5000:
                waveform_1 = extend_array(waveform, ann['SampleMark'][0]['startSample'],
                                          ann['SampleMark'][0]['endSample'], 5000)
            if 2500 < len_sample_mark2 < 5000:
                waveform_2 = extend_array(waveform, ann['SampleMark2'][0]['startSample'],
                                          ann['SampleMark2'][0]['endSample'], 5000)

            if len(waveform_2) == 5000:
                waveform = waveform_2
            elif len(waveform_1) == 5000:
                waveform = waveform_1
            else:
                if ann['SampleMark'][0]['startSample'] < ann['SampleMark2'][0]['startSample']:
                    waveform = np.concatenate((waveform_1, waveform_2))
                else:
                    waveform = np.concatenate((waveform_2, waveform_1))

        waveform = np.expand_dims(waveform, axis=-1)
        waveform = np.nan_to_num(waveform)
        spec = np.squeeze(waveform)

        return waveform, spec, idx

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item


def get_loaders(params, topic):
    train_df = pd.read_csv(params['train_labels_csv'])
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(params['val_labels_csv'])
    val_df = val_df.sample(frac=1).reset_index(drop=True)

    is_train, vocab = True, None
    trainset = RealDataset(len(train_df), topic, vocab, is_train, params['data_dir'], params['in_length'],
                           params['num_classes'], train_df, batch_size=params['batch_size'])

    is_train, vocab = False, trainset.vocab
    valset = RealDataset(len(val_df), topic, vocab, is_train, params['data_dir'], params['in_length'],
                         params['num_classes'], val_df, batch_size=params['batch_size'])

    test_df = pd.read_csv(params['test_labels_csv'])
    testset = RealDataset(len(test_df), topic, vocab, is_train, params['data_dir'], params['in_length'],
                          params['num_classes'], test_df, batch_size=params['batch_size'])

    return trainset, valset, testset, vocab


def extend_array(general_array, start, end, k):
    len_extend_array = end - start + 1
    while len_extend_array < k:
        if start:
            start -= 1
        if end != len(general_array) - 1:
            end += 1
        len_extend_array = end - start + 1
    return general_array[start:end + 1]
