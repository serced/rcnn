
import gzip
import random
import json
import os
import cPickle as pickle

import theano
import numpy as np

from nn import EmbeddingLayer
from utils import say, load_embedding_iterator

from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
detok = TreebankWordDetokenizer()

def read_rationales(path):
    data = [ ]
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            item = json.loads(line)
            data.append(item)
    return data

def read_annotations(path):
    data_x, data_y = [ ], [ ]
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            y, sep, x = line.partition("\t")
            x, y = x.split(), y.split()
            if len(x) == 0: continue
            y = np.asarray([ float(v) for v in y ], dtype = theano.config.floatX)
            data_x.append(x)
            data_y.append(y)
    say("{} examples loaded from {}\n".format(
            len(data_x), path
        ))
    say("max text length: {}\n".format(
        max(len(x) for x in data_x)
    ))
    return data_x, data_y

def get_reviews(data_dir, data_name, split="train"):
    """ import the rotten tomatoes movie review dataset
    Args:
        data_dir (str): path to directory containing the data files
        data_name (str): name of the data files
        split (str "train"): data split
    Returns:
        features and labels
    """
    assert split in [
        'train', 'val', 'test'], "Split not valid, has to be 'train', 'val', or 'test'"
    split = "dev" if split == "val" else split

    text, labels = [], []

    set_dir = os.path.join(data_dir, data_name, split)
    text_tmp = pickle.load(
        open(os.path.join(set_dir, 'word_sequences') + '.pkl', 'rb'))
    # join tokenized sentences back to full sentences for sentenceBert
    # text_tmp = [detok.detokenize(sub_list) for sub_list in text_tmp]
    text.append(text_tmp)
    label_tmp = pickle.load(
        open(os.path.join(set_dir, 'labels') + '.pkl', 'rb'))
    # convert 'pos' & 'neg' to 1 & 0
    print split
    # label_tmp = convert_label(label_tmp)
    labels.append(label_tmp)
    return text[0], labels[0]

def convert_label(labels):
    """ Convert str labels into integers.
    Args:
        labels (Sequence): list of labels
    returns
        converted labels with integer mapping
    """
    converted_labels = []
    print labels
    for i, label in enumerate(labels):
        if label == 'pos':
            # it will be subtracted by 1 in hte label pipeline
            converted_labels.append(2)
        elif label == 'neg':
            converted_labels.append(1)
    return converted_labels

def create_embedding_layer(path):
    embedding_layer = EmbeddingLayer(
            n_d = 200,
            vocab = [ "<unk>", "<padding>" ],
            embs = load_embedding_iterator(path),
            oov = "<unk>",
            #fix_init_embs = True
            fix_init_embs = False
        )
    return embedding_layer


def create_batches(x, y, batch_size, padding_id, sort=True):
    batches_x, batches_y = [ ], [ ]
    N = len(x)
    M = (N-1)/batch_size + 1
    # print y
    if sort:
        perm = range(N)
        perm = sorted(perm, key=lambda i: len(x[i]))
        x = [ x[i] for i in perm ]
        y = [ y[i] for i in perm ]
    for i in xrange(M):
        bx, by = create_one_batch(
                    x[i*batch_size:(i+1)*batch_size],
                    y[i*batch_size:(i+1)*batch_size],
                    padding_id
                )
        batches_x.append(bx)
        batches_y.append(by)
    if sort:
        random.seed(5817)
        perm2 = range(M)
        random.shuffle(perm2)
        batches_x = [ batches_x[i] for i in perm2 ]
        batches_y = [ batches_y[i] for i in perm2 ]
    return batches_x, batches_y

def create_one_batch(lstx, lsty, padding_id):
    max_len = max(len(x) for x in lstx)
    assert min(len(x) for x in lstx) > 0
    bx = np.column_stack([ np.pad(x, (max_len-len(x),0), "constant",
                        constant_values=padding_id) for x in lstx ])
    by = np.vstack(lsty).astype(theano.config.floatX)
    return bx, by
