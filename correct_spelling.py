import sys
import os
import time

import copy
import csv
from itertools import groupby

import numpy as np
import pandas as pd
import re
import json

from cdifflib import CSequenceMatcher
from multiprocessing import Pool

from eval import create_phrase_index

sys.path.append('..')

dictionary_general = ['a', 'aberration', 'accelerated', 'advanced', 'and', 'arrhythmia', 'artifact', 'at', 'atrial',
                      'auricular', 'av', 'any',
                      'baseline', 'beats', 'bigeminy', 'biphasic', 'block', 'bpm', 'bradycardia', 'breath', 'business',
                      'by',
                      'call', 'cardiology', 'care', 'cct', 'cdt', 'cell', 'changes', 'clinic', 'comments', 'comstock',
                      'conduction', 'contact', 'converting', 'correction', 'connected', 'couplets', 'course',
                      'criteria', 'cst', 'complete',
                      'date', 'dizziness', 'define', 'degree', 'detected', 'difficult', 'directly', 'do', 'dropped',
                      'dual', 'due', 'duration',
                      'ectopic', 'email', 'emergent', 'enough', 'episodes', 'escape', 'event', 'expire', 'excision',
                      'fc', 'fibrillation', 'fibrillation/flutter', 'followed', 'for', 'from', 'flutter', 'fusion',
                      'grade', 'gradual', 'had', 'hard', 'have', 'heart', 'high', 'hours',
                      'id', 'idioventricular', 'in', 'including', 'intermittent', 'interpolated', 'interval', 'into',
                      'invalid', 'inverted', 'ivcd',
                      'junctional', 'just',
                      'lasted', 'leads', 'left', 'less', 'longest', 'loss',
                      'max', 'message', 'minutes', 'more', 'morphology', 'multifocal',
                      'need', 'new', 'no', 'nonconducted', 'noted', 'notification', 'notify', 'number', 'nurse', 'nd',
                      'of', 'off', 'office', 'offset', 'on', 'onset', 'operator', 'other',
                      'pac', 'pacs', 'paced', 'pacemaker', 'paroxysmal', 'pause', 'pjc', 'pjcs', 'polymorphic',
                      'possible', 'p', 'pre', 'pvt',
                      'patient', 'phone', 'physician', 'polymorphic', 'possible', 'posted', 'practitioner',
                      'progressing', 'prolonged', 'psvt', 'pvc', 'pvcs', 'pve',
                      'qrs', 'qt', 'quadrigeminy', 'questions',
                      'ranging', 'rate', 'reason', 'reported', 'reversed', 'rhythm', 'run', 'rb', 'rd',
                      'see', 'second', 'seconds', 'send', 'shortest', 'shows', 'single', 'sinus', 'spoke', 'standstill',
                      'stated', 'study', 'support', 'supraventricular', 'sve', 'sves', 'svt', 'symptomatic', 'st',
                      'technician', 'tachycardia', 'terminating', 'text', 'than', 'the', 'then', 'time', 'to', 'took',
                      'trigeminy', 'type', 'triplet',
                      'up', 'urgent',
                      'variable', 've', 'ventricular', 'ves', 'vs', 'view', 'voicemail',
                      'was', 'wave', 'will', 'with',
                      '<unk>']

dictionary = {'a': ['a', 'aberration', 'accelerated', 'advanced', 'and', 'arrhythmia', 'artifact', 'at', 'atrial',
                    'auricular', 'av', 'any'],
              'b': ['baseline', 'beats', 'bigeminy', 'biphasic', 'block', 'bpm', 'bradycardia', 'breath', 'business',
                    'by'],
              'c': ['call', 'cardiology', 'care', 'cct', 'cdt', 'cell', 'changes', 'clinic', 'comments', 'comstock',
                    'conduction', 'contact', 'converting', 'correction', 'connected', 'couplets', 'course', 'criteria',
                    'cst', 'complete'],
              'd': ['date', 'dizziness', 'define', 'degree', 'detected', 'difficult', 'directly', 'do', 'dropped',
                    'dual', 'due', 'duration'],
              'e': ['ectopic', 'email', 'emergent', 'enough', 'episodes', 'escape', 'event', 'expire', 'excision'],
              'f': ['fc', 'fibrillation', 'fibrillation/flutter', 'followed', 'for', 'from', 'flutter', 'fusion'],
              'g': ['grade', 'gradual'],
              'h': ['had', 'hard', 'have', 'heart', 'high', 'hours'],
              'i': ['id', 'idioventricular', 'in', 'including', 'intermittent', 'interpolated', 'interval', 'into',
                    'invalid', 'inverted', 'ivcd'],
              'j': ['junctional', 'just'],
              'l': ['lasted', 'leads', 'left', 'less', 'longest', 'loss'],
              'm': ['max', 'message', 'minutes', 'more', 'morphology', 'multifocal'],
              'n': ['need', 'new', 'no', 'nonconducted', 'noted', 'notification', 'notify', 'number', 'nurse', 'nd'],
              'o': ['of', 'off', 'office', 'offset', 'on', 'onset', 'operator', 'other'],
              'p': ['pac', 'pacs', 'paced', 'pacemaker', 'paroxysmal', 'pause', 'pjc', 'pjcs', 'polymorphic',
                    'possible', 'p', 'pre', 'pvt', 'patient', 'phone', 'physician', 'polymorphic', 'possible', 'posted',
                    'practitioner', 'progressing', 'prolonged', 'psvt', 'pvc', 'pvcs', 'pve'],
              'q': ['qrs', 'qt', 'quadrigeminy', 'questions'],
              'r': ['ranging', 'rate', 'reason', 'reported', 'reversed', 'rhythm', 'run', 'rb', 'rd'],
              's': ['see', 'second', 'seconds', 'send', 'shortest', 'shows', 'single', 'sinus', 'spoke', 'standstill',
                    'stated', 'study', 'support', 'supraventricular', 'sve', 'sves', 'svt', 'symptomatic', 'st'],
              't': ['technician', 'tachycardia', 'terminating', 'text', 'than', 'the', 'then', 'time', 'to', 'took',
                    'trigeminy', 'type', 'triplets'],
              'u': ['up', 'urgent'],
              'v': ['variable', 've', 'ventricular', 'ves', 'vs', 'view', 'voicemail'],
              'w': ['was', 'wave', 'will', 'with'],
              '<': ['<unk>']}

dictionary_train = ['1st', '2nd', '3rd', '1', '2',
                    'and', 'arrhythmia', 'artifact', 'at', 'atrial', 'av',
                    'beats', 'block', 'bpm', 'bradycardia', 'bigeminy',
                    'conduction', 'couplets',
                    'degree',
                    'fibrillation/flutter',
                    'ivcd',
                    'offset', 'onset', 'of',
                    'pause', 'pacs', 'paced', 'pvcs',
                    'quadrigeminy',
                    'rhythm', 'run',
                    'seconds', 'sinus', 'svt', 'sves', 'supraventricular',
                    'tachycardia', 'type', 'trigeminy', 'triplets',
                    'urgent',
                    'vt', 'ves', 'ventricular',
                    'with',
                    '<unk>']


def match_and_split(corpus_splitting_k, index_word_chosen, dictionary, dictionary_general, new_labels, list_unk_num):
    origin_labels = copy.deepcopy(new_labels)
    if dictionary_general[index_word_chosen] in corpus_splitting_k \
            and len(corpus_splitting_k) > len(dictionary_general[index_word_chosen]):
        remain_word = corpus_splitting_k.replace(dictionary_general[index_word_chosen], '')
        if len(remain_word):
            remain_word_pro = map_to_dictionary(corpus_splitting_k.replace(dictionary_general[index_word_chosen], ''),
                                                dictionary, dictionary_general, type='match_and_split')
        else:
            remain_word_pro = np.zeros(len(dictionary_general))
            remain_word_pro[index_word_chosen] = 1
        if dictionary_general[np.argmax(remain_word_pro)] in remain_word:
            if dictionary_general[index_word_chosen] == corpus_splitting_k[:len(dictionary_general[index_word_chosen])]:
                new_labels = new_labels + dictionary_general[index_word_chosen] + ' '
                if len(remain_word) >= len(dictionary_general[np.argmax(remain_word_pro)]):
                    new_labels, list_unk_num = match_and_split(remain_word, np.argmax(remain_word_pro), dictionary,
                                                               dictionary_general,
                                                               new_labels, list_unk_num)
            else:
                if len(remain_word) >= len(dictionary_general[np.argmax(remain_word_pro)]):
                    new_labels, list_unk_num = match_and_split(remain_word, np.argmax(remain_word_pro), dictionary,
                                                               dictionary_general,
                                                               new_labels, list_unk_num)
                new_labels = new_labels + dictionary_general[index_word_chosen] + ' '

        else:
            new_labels = new_labels + dictionary_general[index_word_chosen] + ' '
    else:
        new_labels = new_labels + dictionary_general[index_word_chosen] + ' '
    if '<unk>' in corpus_splitting_k and '<unk>' not in new_labels[len(origin_labels):]:
        new_labels_splitting = new_labels.split(' ')
        unk_index = np.where(np.array(new_labels_splitting) == '<unk>')[0]
        if len(unk_index):
            del list_unk_num[len(unk_index)]
        else:
            del list_unk_num[0]

    return new_labels, list_unk_num


def map_to_dictionary(analyzed_word, dictionary, dictionary_general, type=None):
    first_letter = analyzed_word[0]
    try:
        dictionary_first = dictionary[first_letter]
    except:
        return np.zeros(len(dictionary_general))
    if type is None:
        word_pro = np.zeros(len(dictionary_first))
        for m in range(0, len(dictionary_first)):
            s = CSequenceMatcher(None, analyzed_word, dictionary_first[m]).ratio()
            word_pro[m] = copy.deepcopy(s)
        word_pro_general = np.zeros(len(dictionary_general))
        po_fist = np.where(np.array(dictionary_general) == dictionary_first[0])[0][0]
        word_pro_general[po_fist:po_fist + len(dictionary_first)] = word_pro
        return word_pro_general
    else:
        word_pro_general = np.zeros(len(dictionary_general))
        po_fist = np.where(np.array(dictionary_general) == dictionary_first[0])[0][0]
        for i in range(0, len(dictionary_first)):
            if dictionary_first[i] in analyzed_word:
                word_pro_general[po_fist + i] = 1
                break
        return word_pro_general


def remove_duplicate(corpus_splitting, list_unk_num):
    urgent_index = np.where((np.array(corpus_splitting) == 'urgent') | (np.array(corpus_splitting) == 'emergent'))[0]
    if len(urgent_index):
        corpus_splitting = corpus_splitting[urgent_index[0]:]
        count_unk = len(np.where(np.array(corpus_splitting) == '<unk>')[0])
        list_unk_num = list_unk_num[len(list_unk_num) - count_unk:]
    corpus_splitting = [i[0] for i in groupby(corpus_splitting)]
    return corpus_splitting, list_unk_num


def process_comments(label_, dictionary, dictionary_general):
    label = label_.replace('type ii', 'type 2')
    label = label.replace('type i', 'type 1')
    label = label.replace('avb', 'av block')
    label = label.replace('first', '1st')
    corpus = label.replace('afib', 'atrial fibrillation')

    new_labels = ''
    corpus = re.sub('[^a-zA-Z0-9\.]', ' ', corpus)

    list_unk_num = re.findall('\d*\.?\d+', corpus)
    corpus = re.sub(r'\d*\.?\d+', '<unk>', corpus)

    corpus = corpus.replace('fibrillation flutter', 'fibrillation/flutter')

    corpus_splitting = corpus.split(' ')

    urgent_index = np.where((np.array(corpus_splitting) == 'urgent') | (np.array(corpus_splitting) == 'emergent'))[0]
    if len(urgent_index):
        corpus_splitting = corpus_splitting[urgent_index[0]:]

    while '' in corpus_splitting:
        corpus_splitting.remove('')
    while '\\' in corpus_splitting:
        corpus_splitting.remove('\\')
    k = 0
    while k < len(corpus_splitting):
        max_chose = 0
        analyzed_word = ''
        num_loop = -1
        remember_loop = 0
        while True:
            num_loop += 1
            try:
                analyzed_word = analyzed_word + corpus_splitting[k + num_loop]
            except:
                break
            word_pro = map_to_dictionary(analyzed_word, dictionary, dictionary_general)
            if max_chose <= max(word_pro) and (
                    dictionary_general[np.argmax(word_pro)][0] == corpus_splitting[k][0] or num_loop == 0):
                max_chose = max(word_pro)
                index_word_chosen = int(np.argmax(word_pro))
                remember_loop = copy.deepcopy(num_loop)
            else:
                break

        new_labels, list_unk_num = match_and_split(corpus_splitting[k], index_word_chosen, dictionary,
                                                   dictionary_general, new_labels, list_unk_num)
        k += remember_loop + 1
    new_corpus_splitting = new_labels.split(' ')
    new_corpus_splitting, list_unk_num = remove_duplicate(new_corpus_splitting, list_unk_num)
    new_labels = ' '.join(new_corpus_splitting)
    return new_labels, list_unk_num


def change_special_num(spelling_comment, list_unk_num):
    corpus_splitting = spelling_comment.split(' ')
    list_unk = np.where(np.array(corpus_splitting) == '<unk>')[0]
    list_unk_before = list_unk - 1
    list_unk_before = np.where(list_unk_before < 0, 0, list_unk_before)
    type_index = np.where(np.array(np.array(corpus_splitting)[list_unk_before]) == 'type')[0]
    training_comment = copy.deepcopy(spelling_comment)
    if '<unk> st' in training_comment:
        training_comment = training_comment.replace('<unk> st', '1st')
    if '<unk> nd' in training_comment:
        training_comment = training_comment.replace('<unk> nd', '2nd')
    if '<unk> rd' in training_comment:
        training_comment = training_comment.replace('<unk> rd', '3rd')
    if 'type <unk>' in training_comment:
        training_comment = training_comment.replace('type <unk>', f'type {list_unk_num[type_index[0]]}')

    return training_comment


def correct_comment_phase1(label):
    try:
        label = label.replace(';', ' ')
        label = label.replace('\t', ' ')
        label = label.lower()
        spelling_comment, list_unk_num = process_comments(label, dictionary, dictionary_general)

        if len(spelling_comment.split()) == 1 and spelling_comment.split()[0] not in ['artifact', 'afib']:
            return None
        training_comment = change_special_num(spelling_comment, list_unk_num)

        return training_comment

    except:
        return None


def correct_comment_phase2(label):
    if label is None:
        return None
    if 'study id <unk>' in label:
        labels_repair = label.replace('study id <unk>', '')
    else:
        labels_repair = label
    if 'event id <unk>' in labels_repair:
        labels_repair = labels_repair.replace('event id <unk>', '')
    if 'second degree' in labels_repair:
        labels_repair = labels_repair.replace('second degree', '2nd degree')
    if 'paroxysmal supraventricular tachycardia' in labels_repair:
        labels_repair = labels_repair.replace('paroxysmal supraventricular tachycardia', 'svt')
    if 'supraventricular tachycardia' in labels_repair:
        labels_repair = labels_repair.replace('supraventricular tachycardia', 'svt')
    if 'psvt' in labels_repair:
        labels_repair = labels_repair.replace('psvt', 'svt')
    if 'polymorphic ventricular tachycardia' in labels_repair:
        labels_repair = labels_repair.replace('polymorphic ventricular tachycardia', 'vt')
    if 'ventricular tachycardia' in labels_repair:
        labels_repair = labels_repair.replace('ventricular tachycardia', 'vt')
    if 'pvt' in labels_repair:
        labels_repair = labels_repair.replace('pvt', 'vt')
    if 'ventricular run' in labels_repair:
        labels_repair = labels_repair.replace('ventricular run', 'vt')
    if 'fibrillation' in labels_repair and 'fibrillation/flutter' not in labels_repair:
        labels_repair = labels_repair.replace('fibrillation', 'fibrillation/flutter')
    if 'flutter' in labels_repair and 'fibrillation/flutter' not in labels_repair:
        labels_repair = labels_repair.replace('flutter', 'fibrillation/flutter')
    if 'advanced heart' in labels_repair:
        labels_repair = labels_repair.replace('advanced heart', '3rd degree av')
    if 'couplets heart' in labels_repair:
        labels_repair = labels_repair.replace('couplets heart', '3rd degree av')
    if 'complete heart' in labels_repair:
        labels_repair = labels_repair.replace('complete heart', '3rd degree av')
    if 'degree heart' in labels_repair:
        labels_repair = labels_repair.replace('degree heart', 'degree av')
    if 'emergent' in labels_repair:
        labels_repair = labels_repair.replace('emergent', 'urgent')

    corpus_split = np.array(labels_repair.split())
    if 'pac' in corpus_split:
        corpus_split[corpus_split == 'pac'] = 'pacs'
    if 've' in corpus_split:
        corpus_split[corpus_split == 've'] = 'ves'
    if 'pacemaker' in corpus_split:
        corpus_split[corpus_split == 'pacemaker'] = 'paced'
    if 'pvc' in corpus_split:
        corpus_split[corpus_split == 'pvc'] = 'pvcs'
    if 'sve' in corpus_split:
        corpus_split[corpus_split == 'sve'] = 'sves'
    if 'second' in corpus_split:
        corpus_split[corpus_split == 'second'] = 'seconds'

    corpus_split_ = list(corpus_split)
    j = 0
    while j < len(corpus_split_):
        if corpus_split_[j] not in dictionary_train:
            corpus_split_.remove(corpus_split_[j])
        else:
            j += 1

    while True:
        if len(corpus_split_) and corpus_split_[-1] in ['and', 'with', 'urgent', 'atrial', 'ventricular',
                                                        'supraventricular', 'at']:
            corpus_split_.pop(-1)
        elif len(corpus_split_) >= 2 and corpus_split_[-1] == '<unk>' and corpus_split_[-2] != 'at':
            corpus_split_.pop(-1)
        else:
            break

    corpus = ' '.join(corpus_split_)
    phrase_index = create_phrase_index(corpus)
    if len(corpus_split_) and not (len(corpus_split_) == 1 and corpus_split_[0] != 'artifact') \
            and np.sum(phrase_index) >= 1:
        if ('2nd' in corpus or '3rd' in corpus) and 'urgent' not in corpus:
            corpus = 'urgent ' + corpus
        if 'at <unk>' in corpus and 'at <unk> bpm' not in corpus:
            corpus = corpus.replace('at <unk>', 'at <unk> bpm')
        if 'at bpm' in corpus and 'at <unk> bpm' not in corpus:
            corpus = corpus.replace('at bpm', 'at <unk> bpm')
        if '<unk> bpm' in corpus and 'at <unk> bpm' not in corpus:
            corpus = corpus.replace('<unk> bpm', 'at <unk> bpm')
        if 'with and' in corpus:
            corpus = corpus.replace('with and', 'with')
        if 'with at' in corpus and 'with atrial' not in corpus:
            corpus = corpus.replace('with at', 'at')
        if 'with beats' in corpus:
            corpus = corpus.replace('with beats', '')
        if 'and beats' in corpus:
            corpus = corpus.replace('and beats', '')
        if 'bpm beats' in corpus:
            corpus = corpus.replace('bpm beats', 'bpm')
        if 'run beats' in corpus:
            corpus = corpus.replace('run beats', 'run')
        if 'pause beats' in corpus:
            corpus = corpus.replace('pause beats', 'pause')

        return corpus
    else:
        return None


def correct_comment(label):
    corpus_1 = correct_comment_phase1(label)
    corpus_2 = correct_comment_phase2(corpus_1)
    return corpus_2


if __name__ == '__main__':
    params = json.load(open('config.json', 'r'))

    name_raw_file = params['raw_comments_csv']
    name_processed_file = params['corrected_spelling_comments_csv']

    _df = pd.read_csv(name_raw_file)
    labels = _df['comments']
    ids = _df['id']
    count = 0

    with open(name_processed_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['EventID', 'Raw Labels', 'Label'])

    print('Start correcting spelling...')
    t0 = time.time()
    with Pool(processes=os.cpu_count()) as pool:
        training_comment = pool.map(correct_comment, labels)

    for i in range(len(labels)):
        if training_comment[i] is not None:
            with open(name_processed_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([ids[i], labels[i].replace(';', ' ').replace('\t', ' '), training_comment[i]])
        else:
            print('ID:', ids[i])
            print('Raw Labels:', labels[i])
            print('\n')
            count += 1

    print('Processing Time', time.time() - t0)
    print('Num lost:', count)
    print('Correct Spelling: Done')
