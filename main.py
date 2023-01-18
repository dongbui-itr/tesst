import json
import os.path

from numba import cuda
from datetime import datetime

import tensorflow as tf
from util import get_loaders
from topic_transformer import TopicTransformer, CustomCallback
import pickle
from ecg_resnet import ECGResNet, BasicBlock
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers


def cli_main(params, dev):
    model_name = params['model_name']
    now = datetime.now()
    dt_string = now.strftime("%Y:%m:%d-%H:%M:%S")
    save_path = f'training/tf/model_{model_name}_epoch={params["epochs"]}_time={dt_string}'
    model_path = save_path + '/model'
    checkpoint_path = save_path + '/checkpoints'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    topic = True
    train_loader, val_loader, test_loader, vocab = get_loaders(params, topic=topic)

    with open(save_path + '/vocab.pkl', 'wb') as file:
        pickle.dump(vocab, file)

    model = TopicTransformer(vocab, **params)
    model.build(input_shape=[(params['batch_size'], params['in_length'], params['in_channels']),
                             (params['batch_size'], 128, 30, 1),
                             (params['batch_size'], params['max_length'])])
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, run_eagerly=True)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, 'weights-{epoch:02d}.pb'),
        save_weights_only=True,
        mode='max'
    )

    history = model.fit(train_loader,
                        epochs=params['epochs'],
                        validation_data=val_loader,
                        batch_size=params['batch_size'],
                        callbacks=[model_checkpoint_callback, CustomCallback()])

    # save model
    with open(model_path + '/history.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    model.save(model_path, save_format='tf')


def take_model_shape(params):
    topic = True
    train_loader, val_loader, test_loader, vocab = get_loaders(params, topic=topic)
    model_name = params['model_name']
    model = TopicTransformer(vocab, **params)
    model.build(input_shape=[(params['batch_size'], params['in_length'], params['in_channels']),
                             (params['batch_size'], 128, 30, 1),
                             (params['batch_size'], params['max_length'])])

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, run_eagerly=True)


if __name__ == '__main__':
    device = cuda.get_current_device()
    device.reset()

    params, dev = json.load(open('config.json', 'r')), False

    cli_main(params, dev)

    params, dev = json.load(open('config.json', 'r')), False
    take_model_shape(params)

    # Test the model

    # in_length = 5000
    # in_channels = 1
    # n_grps = 3
    # N = 4
    # dropout = 0.3
    # first_width = 32
    # stride = [2, 1, 2, 1]
    # dilation = 1
    # topic = True
    # d_mode = 128
    # train_loader, val_loader, test_loader, vocab = get_loaders(params, topic=topic)
    # rate = 100
    # model = ECGResNet(5000, 1, 1, 4, 2, 16, [2, 1, 2, 1], 1, 100)
    # print(model.summary((5000, 1)))

