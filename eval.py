import numpy as np

from ptbtokenize import PTBTokenizer
from cider import Cider
from bleu import Bleu
from meteor import Meteor
from rouge import Rouge


class COCOEvalCap:
    def __init__(self):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}

    def evaluate(self, gts, res):

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        # print(gts)
        # =================================================
        # Set up scorers
        # =================================================
        # print('----- gts', gts)
        # print('----- res', res)
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            if method == 'METEOR':
                continue
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                print("%s: %0.3f" % (method, score))

    def setEval(self, score, method):
        self.eval[method] = score


def create_phrase_index(corpus):
    list_index = np.zeros(10)
    if 'fibrillation' in corpus:
        list_index[0] = 1
    if '2nd' in corpus:
        list_index[1] = 1
    if '3rd' in corpus:
        list_index[2] = 1
    if 'sinus rhythm' in corpus or 'sinus arrhythmia' in corpus or '1st' in corpus:
        list_index[3] = 1
    if 'svt' in corpus:
        list_index[4] = 1
    if 'vt' in corpus.split():
        list_index[5] = 1
    if 'sinus tachycardia' in corpus:
        list_index[6] = 1
    if 'sinus bradycardia' in corpus:
        list_index[7] = 1
    if 'pause' in corpus:
        list_index[8] = 1
    if 'artifact' in corpus:
        list_index[9] = 1
    return list_index


def evaluate_new(gts, res):
    list_index_gts = create_phrase_index(gts)
    list_index_res = create_phrase_index(res)
    result = 1
    for i in range(len(list_index_gts)):
        if list_index_gts[i] == 1 and list_index_res[i] == 0:
            result = 0
            break
    return result


def evaluate_for_confusion(gts, res, ignore=None):
    list_index_gts = create_phrase_index(gts)
    list_index_res = create_phrase_index(res)

    # define result matrix with 4 row and 10 columns
    # 4 rows corresponding to TP, FN, FP, TN
    # 10 columns corresponding to types
    sub_confusion_matrix = np.zeros((4, 10))
    sub_confusion_matrix[0, :] = np.multiply(list_index_gts, list_index_res)
    sub_confusion_matrix[1, :] = np.multiply(list_index_gts, np.ones(10) - list_index_res)
    sub_confusion_matrix[2, :] = np.multiply(list_index_res, np.ones(10) - list_index_gts)
    sub_confusion_matrix[3, :] = np.multiply(np.ones(10) - list_index_gts, np.ones(10) - list_index_res)

    if ignore == 'sinus':
        if ('sinus arrhythmia' in gts and 'sinus arrhythmia' in res) or ('1st' in gts and '1st' in res):
            pass
        elif list_index_gts[3] == 1 and np.sum(list_index_gts) >= 2 \
                and list_index_res[3] == 1 and np.sum(list_index_res) == 1:
            sub_confusion_matrix[0, 3] = 0
            sub_confusion_matrix[1, 3] = 1
    check_predict = 1

    for i in range(len(list_index_gts)):
        if list_index_gts[i] == 1 and list_index_res[i] == 0:
            check_predict = 0
            break

    return sub_confusion_matrix, check_predict
