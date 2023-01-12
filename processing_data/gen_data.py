import os
import sys
import time

import numpy as np
import pandas as pd
import csv
import json
import wfdb
from multiprocessing import Pool

sys.path.append('..')


def aggregate_data(event_id, comment):
    try:
        if os.path.exists(base_data_path + '/' + event_id + '.rev'):
            with open(base_data_path + '/' + event_id + '.rev') as f:
                data = f.read()
            f.close()
        else:
            with open(base_data_path + '/' + event_id + '.rhy') as f:
                data = f.read()
            f.close()
    except:
        return None, None, None

    if data:
        ann = json.loads(data)
        study_id = ann['studyFid']

        return event_id, comment, study_id

    return None, None, None


def get_all_index_of_study_id(study_id):
    return np.where(aggregated_data[:, 2] == str(study_id))[0]


def get_data_idx_from_study_id(study_ids):
    idx = []
    with Pool(processes=os.cpu_count()) as pool:
        try:
            for inds in pool.map(get_all_index_of_study_id, study_ids):
                idx = inds if not len(idx) else np.append(idx, inds)
        except Exception as e:
            print(e)

    return idx


def get_channel(event_id, comment, study_id):
    try:
        raw_signal, fields = wfdb.rdsamp(os.path.join(base_data_path, event_id))
        if len(raw_signal) != 15000:
            return None, None, None, None

        if 'artifact' in comment:
            channel = 1
        else:
            if os.path.exists(os.path.join(base_data_path, event_id) + '.rev'):
                with open(os.path.join(base_data_path, event_id) + '.rev') as f1:
                    s = f1.read()
                f1.close()
            else:
                with open(os.path.join(base_data_path, event_id) + '.rhy') as f1:
                    s = f1.read()
                f1.close()
            if not s:
                return None, None, None, None

            ann = json.loads(s)
            if len(ann['SampleMark']):
                channel = ann['SampleMark'][0]['channelIndex']
            elif len(ann['SampleMark2']):
                channel = ann['SampleMark2'][0]['channelIndex']
            else:
                return None, None, None, None

            rest_channels = [0, 1, 2]
            rest_channels.remove(channel)
            if not np.any(raw_signal[:, channel]):
                channel = rest_channels[0]
                if not np.any(raw_signal[:, channel]):
                    channel = rest_channels[0]
                    if not np.any(raw_signal[:, channel]):
                        channel = rest_channels[1]
                    else:
                        return None, None, None, None

        return event_id, comment, channel, study_id
    except:
        return None, None, None, None


def gen_data_to_file(path, name, list_idx):
    print(f'Start generating {name} data...')
    with open(path, 'w') as f:
        writer = csv.writer(f)

        headers = np.array(['EventID', 'Label', 'Channel', 'StudyFid'])
        writer.writerow(headers)

        arg_list = [(aggregated_data[idx]) for idx in list_idx]
        with Pool(processes=os.cpu_count()) as pool:
            try:
                for event_id, comment, channel, study_id in pool.starmap(get_channel, arg_list):
                    if event_id and comment and channel and study_id:
                        contents = np.array([event_id, comment, channel, study_id])
                        writer.writerow(contents)
            except Exception as e:
                print(e)

    f.close()
    print('Done')


if __name__ == '__main__':
    params = json.load(open('config.json', 'r'))

    base_data_path = params['data_dir']
    events_csv_path = params['corrected_spelling_comments_csv']
    gen_caption_csv_path_train = params['train_labels_csv']
    gen_caption_csv_path_val = params['val_labels_csv']
    gen_caption_csv_path_test = params['test_labels_csv']

    df = pd.read_csv(events_csv_path)
    event_ids = np.array(df['EventID'].tolist())
    comments = np.array(df['Label'].tolist())
    splitting_rate = 0.8

    print('Start splitting dataset...')
    t0 = time.time()
    aggregated_data = []
    study_ids = []
    arg_list = [(event_ids[i], comments[i]) for i in range(len(event_ids))]
    with Pool(processes=os.cpu_count()) as pool:
        try:
            for event_id, comment, study_id in pool.starmap(aggregate_data, arg_list):
                if event_id and comment and study_id:
                    study_ids.append(study_id)
                    data_row = np.array([event_id, comment, study_id])
                    aggregated_data.append(data_row)
        except Exception as e:
            print(e)
    study_ids = np.unique(study_ids)
    aggregated_data = np.array(aggregated_data)

    inds = np.arange(len(study_ids))
    np.random.shuffle(inds)
    study_ids = study_ids[inds]

    inds = np.arange(len(aggregated_data))
    np.random.shuffle(inds)
    aggregated_data = aggregated_data[inds]

    mark = int(len(study_ids) * splitting_rate)
    train_study_ids, test_study_ids = study_ids[:mark], study_ids[mark:]

    mark = int(len(train_study_ids) * splitting_rate)
    train_study_ids, val_study_ids = train_study_ids[:mark], train_study_ids[mark:]

    train_idx = get_data_idx_from_study_id(train_study_ids)
    val_idx = get_data_idx_from_study_id(val_study_ids)
    test_idx = get_data_idx_from_study_id(test_study_ids)

    gen_data_to_file(path=gen_caption_csv_path_train, name='train', list_idx=train_idx)
    gen_data_to_file(path=gen_caption_csv_path_val, name='val', list_idx=val_idx)
    gen_data_to_file(path=gen_caption_csv_path_test, name='test', list_idx=test_idx)

    print('Processing Time', time.time() - t0)
    print('Splitting dataset: Done')
