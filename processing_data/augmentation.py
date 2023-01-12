import sys

import csv
import json

import pandas as pd

sys.path.append('..')


if __name__ == '__main__':
    params = json.load(open('config.json', 'r'))

    augmented_file = params['augmented_file_csv']
    df = pd.read_csv(params['file_need_to_augment_csv'])

    with open(augmented_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['EventID', 'Label', 'Channel', 'StudyFid', 'IsAugment'])
    for i in range(len(df)):
        label = df['Label'][i]

        if '2nd' in label:
            n_aug = 6
        elif '3rd' in label:
            n_aug = 25
        elif 'svt' in label:
            n_aug = 2
        elif 'vt' in label.split():
            n_aug = 8
        elif 'pause' in label:
            n_aug = 7
        else:
            n_aug = 1

        for j in range(n_aug):
            with open(augmented_file, 'a') as f:
                writer = csv.writer(f)
                if j == 0:
                    is_augment = 0
                else:
                    is_augment = 1
                writer.writerow([df['EventID'][i],
                                 label,
                                 df['Channel'][i],
                                 df['StudyFid'][i],
                                 is_augment])
