"""This file contains code to read tokenized data from file,
truncate, pad and process it into batches ready for training"""

from __future__ import absolute_import
from __future__ import division

import random
import time
import re

import numpy as np
from six.moves import xrange
from vocab import PAD_ID, UNK_ID


class Batch(object):
    """A class to hold the information needed for a training batch"""

    def __init__(self, context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens, ans_span, ans_tokens, uuids=None):
        self.context_ids = context_ids
        self.context_mask = context_mask
        self.context_tokens = context_tokens

        self.qn_ids = qn_ids
        self.qn_mask = qn_mask
        self.qn_tokens = qn_tokens

        self.ans_span = ans_span
        self.ans_tokens = ans_tokens

        self.uuids = uuids

        self.batch_size = len(self.context_tokens)


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def intstr_to_intlist(string):
    """Given a string, returns as a list of integers"""
    return [int(s) for s in string.split()]


def sentence_to_token_ids(sentence, word2id):
    """Turns an already-tokenized sentence string into word indices
    """
    tokens = split_by_whitespace(sentence)
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    return tokens, ids


def padded(token_batch, batch_pad=0):
    
    maxlen = max(map(lambda x: len(x), token_batch)) if batch_pad == 0 else batch_pad
    return list(map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), token_batch))


def refill_batches(batches, word2id, context_file, qn_file, ans_file, batch_size, context_len, question_len, discard_long):
    """
    Adds more batches into the "batches" list.
    """
    print ("Refilling batches...")
    tic = time.time()
    examples = []
    context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()

    while context_line and qn_line and ans_line: 

        # Convert tokens to word ids
        context_tokens, context_ids = sentence_to_token_ids(context_line, word2id)
        qn_tokens, qn_ids = sentence_to_token_ids(qn_line, word2id)
        ans_span = intstr_to_intlist(ans_line)

        # read the next line from each file
        context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()

        # get ans_tokens from ans_span
        assert len(ans_span) == 2
        if ans_span[1] < ans_span[0]:
            print ("Found an ill-formed gold span: start=%i end=%i" % (ans_span[0], ans_span[1]))
            continue
        ans_tokens = context_tokens[ans_span[0] : ans_span[1]+1]

        # discard or truncate too-long questions
        if len(qn_ids) > question_len:
            if discard_long:
                continue
            else:
                qn_ids = qn_ids[:question_len]

        # discard or truncate too-long contexts
        if len(context_ids) > context_len:
            if discard_long:
                continue
            else:
                context_ids = context_ids[:context_len]

        # add to examples
        examples.append((context_ids, context_tokens, qn_ids, qn_tokens, ans_span, ans_tokens))

        # stop refilling if you have 160 batches
        if len(examples) == batch_size * 160:
            break

    # Once you've either got 160 batches or you've reached end of file:

    # Sort by question length
    # Note: if you sort by context length, then you'll have batches which contain the same context many times (because each context appears several times, with different questions)
    examples = sorted(examples, key=lambda e: len(e[2]))

    for batch_start in xrange(0, len(examples), batch_size):

        context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch = zip(*examples[batch_start:batch_start+batch_size])

        batches.append((context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch))

    # shuffle the batches
    random.shuffle(batches)

    toc = time.time()
    print ("Refilling batches took %.2f seconds" % (toc-tic))
    return


def get_batch_generator(word2id, context_path, qn_path, ans_path, batch_size, context_len, question_len, discard_long):
    """
    This function returns a generator object that yields batches.
    The last batch in the dataset will be a partial batch.
    Read this to understand generators and the yield keyword in Python: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    """
    context_file, qn_file, ans_file = open(context_path,encoding="utf-8"), open(qn_path,encoding="utf-8"), open(ans_path,encoding="utf-8")
    batches = []

    while True:
        if len(batches) == 0: 
            refill_batches(batches, word2id, context_file, qn_file, ans_file, batch_size, context_len, question_len, discard_long)
        if len(batches) == 0:
            break

        # Get next batch. These are all lists length batch_size
        (context_ids, context_tokens, qn_ids, qn_tokens, ans_span, ans_tokens) = batches.pop(0)

        # Pad context_ids and qn_ids
        qn_ids = padded(qn_ids, question_len) 
        context_ids = padded(context_ids, context_len) 

        # Make qn_ids into a np array and create qn_mask
        qn_ids = np.array(qn_ids)
        qn_mask = (qn_ids != PAD_ID).astype(np.int32) 

        # Make context_ids into a np array and create context_mask
        context_ids = np.array(context_ids) 
        context_mask = (context_ids != PAD_ID).astype(np.int32) 

        # Make ans_span into a np array
        ans_span = np.array(ans_span) 

        # Make into a Batch object
        batch = Batch(context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens, ans_span, ans_tokens)

        yield batch

    return
