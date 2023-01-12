import ast
import pickle
import sys
import random

import tqdm

import json
import pandas as pd
import csv

import numpy as np
import tensorflow as tf
from eval import evaluate_for_confusion, COCOEvalCap
from util import RealDataset
from utils_model import get_next_word
from topic_transformer import TopicTransformer
import matplotlib.pyplot as plt


def sample(model, vocab, waveforms, specs, sample_method, max_length):
    nb_batch = waveforms.shape[0]
    waveforms = tf.cast(waveforms, tf.float32)

    image_features, avg_features = model.ecg_resnet_model(waveforms)
    image_features = model.feature_embedding(image_features)

    start_tokens = np.array([vocab('<start>')])
    start_tokens = np.tile(start_tokens, (nb_batch, 1))
    start_tokens = tf.convert_to_tensor(start_tokens)
    sent = model.embed(start_tokens)

    tgt_mask = tf.zeros((sent.shape[0], 1), dtype=bool)
    y_out = tf.zeros([nb_batch, max_length]).numpy()

    for i in range(max_length):
        out, _ = model.transformer(image_features, avg_features, sent, tf.cast(sent[:, :, i], dtype=tf.int64), training=False)
        out = model.to_vocab(out[:, i, :])
        s = sample_method
        word_idx, props = get_next_word(out, temp=s['temp'], k=s['k'], p=s['p'], greedy=s['greedy'], m=s['m'])
        y_out[:, i] = word_idx.numpy()[:, 0]

        if i < max_length - 1:
            tgt_mask = tgt_mask | (word_idx == vocab('<end>'))

            embedded = model.embed(word_idx)
            sent = tf.concat([sent, embedded], 1)
            if tf.reduce_sum(tf.cast(tgt_mask, tf.int32)) == nb_batch:
                break
        else:
            break

    return tf.convert_to_tensor(y_out)


if __name__ == '__main__':
    topic = True

    checkpoint_loc, param_file = 'training/tf/model_topic_fnet_epoch=5_time=2022:08:30-11:41:56', 'config.json'
    log_predict_file = 'results/tf/predict_test.csv'
    log_metric_file = 'results/tf/metric_test.txt'
    log_cf_file = 'results/tf/confusion_matrix.csv'

    params = json.load(open(param_file, 'r'))

    vocab = pickle.load(open(checkpoint_loc + '/vocab.pkl', 'rb'))
    model = TopicTransformer(vocab, **params)
    model.build(input_shape=[(params['batch_size'], params['in_length'], params['in_channels']),
                             (params['batch_size'], 128, 30, 1),
                             (params['batch_size'], params['max_length'])])
    model.load_weights(checkpoint_loc + '/checkpoints/weights-05.pb')

    is_train = False
    test_df = pd.read_csv(params['test_labels_csv'])
    testset = RealDataset(len(test_df), topic, vocab, is_train, params['data_dir'], params['in_length'],
                          params['num_classes'], test_df, batch_size=params['batch_size'])
    event_ids = test_df['EventID']

    # For transformer topic
    sample_method = {'temp': None, 'k': None, 'p': None, 'greedy': True, 'm': None}
    max_length = params['max_length']

    verbose = False
    gts = {}
    res = {}

    predict_results = []
    confusion_matrix = np.zeros((4, 10))
    with open(log_predict_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['EventID', 'Technician Comments', 'Predict Comments', 'Channel'])
        for batch_idx, batch in enumerate(tqdm.tqdm(testset)):
            waveforms, specs, ids, targets, weights, labels = batch
            ids = ids.numpy()
            words = sample(model, vocab, waveforms, specs, sample_method, max_length)
            generated = vocab.decode(words, skip_first=False)
            truth = vocab.decode(targets)

            for i in range(waveforms.shape[0]):
                res[batch_idx * params['batch_size'] + i] = [generated[i]]
                gts[batch_idx * params['batch_size'] + i] = [truth[i]]
                with open(log_predict_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([event_ids[ids[i]], truth[i], generated[i], test_df['Channel'][i]])
                sub_confusion_matrix, check_predict = evaluate_for_confusion(gts[batch_idx * params['batch_size'] + i][0],
                                                                             res[batch_idx * params['batch_size'] + i][0],
                                                                             ignore='sinus')
                confusion_matrix += sub_confusion_matrix
                predict_results.append(check_predict)
                if verbose and check_predict == 0:
                    print('\n')
                    print(f'True: {gts[batch_idx * params["batch_size"] + i][0]}')
                    print(f'Pred: {res[batch_idx * params["batch_size"] + i][0]}')

    # save confusion matrix
    pd.DataFrame(confusion_matrix).to_csv(log_cf_file)

    print('Percent of wrong sentence is', (1 - np.sum(np.array(predict_results)) / len(predict_results)) * 100, ' %')

    COCOEval = COCOEvalCap()
    COCOEval.evaluate(gts, res)
    print(sample_method, COCOEval.eval)
    with open(log_metric_file, 'w') as f:
        for i in COCOEval.eval:
            s = f'{i} {COCOEval.eval[i]}\n'
            f.write(s)
