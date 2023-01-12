import sys

sys.path.append('..')
from operator import itemgetter

import numpy as np
import tensorflow as tf


def collate_fn(data):
    """Creates mini-batch tensors from the dicts (waveform, samplebase, gain, id, captions).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of dict (waveform, samplebase, gain, id, captions).
        max_length: maximum length of a sentence
    Returns:
    """
    captions = [d['label'] for d in data]
    lengths = [len(cap) for cap in captions]

    if len(lengths) == 1:
        return data[0]['waveform'].unsqueeze(0), data[0]['samplebase'], data[0]['gain'], data[0]['id'], data[0][
            'label'].unsqueeze(0).long(), lengths, data[0]['extra_label'].unsqueeze(0)

    ind = np.argsort(lengths)[::-1]

    lengths = list(itemgetter(*ind)(lengths))
    captions = list(itemgetter(*ind)(captions))

    waveforms = list(itemgetter(*ind)([d['waveform'] for d in data]))
    specs = list(itemgetter(*ind)([d['spec'] for d in data]))
    ids = list(itemgetter(*ind)([d['id'] for d in data]))
    weights = list(itemgetter(*ind)([d['weights'] for d in data]))

    # Merge images (from tuple of 3D tensor to 4D tensor).
    waveforms = np.stack(waveforms, 0)
    specs = np.stack(specs, 0)

    targets = np.zeros((len(captions), max(lengths)), dtype=np.int64)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    if 'extra_label' in data[0]:
        extra_label = list(itemgetter(*ind)([d['extra_label'] for d in data]))
        extra_label = np.stack(extra_label, 0)
        return [tf.convert_to_tensor(waveforms), tf.convert_to_tensor(specs), tf.convert_to_tensor(ids),
                tf.convert_to_tensor(targets), tf.convert_to_tensor(weights), tf.convert_to_tensor(extra_label)]

    return [tf.convert_to_tensor(waveforms), tf.convert_to_tensor(specs), tf.convert_to_tensor(ids),
            tf.convert_to_tensor(targets), tf.convert_to_tensor(weights)]
