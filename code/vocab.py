"""This file contains a function to read the GloVe vectors from file,
and return them as an embedding matrix"""

from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import numpy as np

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1


def get_glove(glove_path, glove_dim):
    """Reads from original GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.
    """

    print ("Loading GLoVE vectors from file: %s" % glove_path)
    vocab_size = int(4e5)

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    word2id = {}
    id2word = {}

    random_init = True
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)

    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    with open(glove_path, 'r',encoding="utf-8") as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if glove_dim != len(vector):
                raise Exception("You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (glove_path, glove_dim))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word2id, id2word
