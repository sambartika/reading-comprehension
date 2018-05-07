"""This code is required for "official_eval" mode in main.py
It provides functions to read a SQuAD json file, use the model to get predicted answers,
and write those answers to another JSON file."""

from __future__ import absolute_import
from __future__ import division

import os
from tqdm import tqdm
import numpy as np
from six.moves import xrange
from nltk.tokenize.moses import MosesDetokenizer

from preprocessing.squad_preprocess import data_from_json, tokenize
from vocab import UNK_ID, PAD_ID
from data_batcher import padded, Batch



def readnext(x):
    """x is a list"""
    if len(x) == 0:
        return False
    else:
        return x.pop(0)



def refill_batches(batches, word2id, qn_uuid_data, context_token_data, qn_token_data, batch_size, context_len, question_len):
    """
    This is similar to refill_batches in data_batcher.py, but instead of reading from (preprocessed) datafiles, it reads from the provided lists
    """
    examples = []

    qn_uuid, context_tokens, qn_tokens = readnext(qn_uuid_data), readnext(context_token_data), readnext(qn_token_data)

    while qn_uuid and context_tokens and qn_tokens:

        context_ids = [word2id.get(w, UNK_ID) for w in context_tokens]
        qn_ids = [word2id.get(w, UNK_ID) for w in qn_tokens]

        if len(qn_ids) > question_len:
            qn_ids = qn_ids[:question_len]
        if len(context_ids) > context_len:
            context_ids = context_ids[:context_len]

        examples.append((qn_uuid, context_tokens, context_ids, qn_ids))

        if len(examples) == batch_size:
            break

        qn_uuid, context_tokens, qn_tokens = readnext(qn_uuid_data), readnext(context_token_data), readnext(qn_token_data)

    for batch_start in xrange(0, len(examples), batch_size):
        uuids_batch, context_tokens_batch, context_ids_batch, qn_ids_batch = zip(*examples[batch_start:batch_start + batch_size])

        batches.append((uuids_batch, context_tokens_batch, context_ids_batch, qn_ids_batch))

    return



def get_batch_generator(word2id, qn_uuid_data, context_token_data, qn_token_data, batch_size, context_len, question_len):
    """
    This is similar to get_batch_generator in data_batcher.py, but with some
    differences.

    """
    batches = []

    while True:
        if len(batches) == 0:
            refill_batches(batches, word2id, qn_uuid_data, context_token_data, qn_token_data, batch_size, context_len, question_len)
        if len(batches) == 0:
            break

        (uuids, context_tokens, context_ids, qn_ids) = batches.pop(0)

        qn_ids = padded(qn_ids, question_len) 
        context_ids = padded(context_ids, context_len) 

        qn_ids = np.array(qn_ids)
        qn_mask = (qn_ids != PAD_ID).astype(np.int32)

        context_ids = np.array(context_ids)
        context_mask = (context_ids != PAD_ID).astype(np.int32)

        batch = Batch(context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens=None, ans_span=None, ans_tokens=None, uuids=uuids)

        yield batch

    return


def preprocess_dataset(dataset):
    qn_uuid_data = []
    context_token_data = []
    qn_token_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing data"):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):

            context = unicode(article_paragraphs[pid]['context']) # string

            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)
            context = context.lower()

            qas = article_paragraphs[pid]['qas']

            for qn in qas:
                question = unicode(qn['question']) 
                question_tokens = tokenize(question)

                question_uuid = qn['id']
                qn_uuid_data.append(question_uuid)
                context_token_data.append(context_tokens)
                qn_token_data.append(question_tokens)

    return qn_uuid_data, context_token_data, qn_token_data


def get_json_data(data_filename):
    """
    Read the contexts and questions from a .json file (like dev-v1.1.json)
    """
    if not os.path.exists(data_filename):
        raise Exception("JSON input file does not exist: %s" % data_filename)

    print ("Reading data from %s..." % data_filename)
    data = data_from_json(data_filename)

    print ("Preprocessing data from %s..." % data_filename)
    qn_uuid_data, context_token_data, qn_token_data = preprocess_dataset(data)

    data_size = len(qn_uuid_data)
    assert len(context_token_data) == data_size
    assert len(qn_token_data) == data_size
    print ("Finished preprocessing. Got %i examples from %s" % (data_size, data_filename))

    return qn_uuid_data, context_token_data, qn_token_data


def generate_answers(session, model, word2id, qn_uuid_data, context_token_data, qn_token_data):
    """
    Given a model, and a set of (context, question) pairs, each with a unique ID,
    use the model to generate an answer for each pair, and return a dictionary mapping
    each unique ID to the generated answer.
    """
    uuid2ans = {} 
    data_size = len(qn_uuid_data)
    num_batches = ((data_size-1) / model.FLAGS.batch_size) + 1
    batch_num = 0
    detokenizer = MosesDetokenizer()

    print ("Generating answers...")

    for batch in get_batch_generator(word2id, qn_uuid_data, context_token_data, qn_token_data, model.FLAGS.batch_size, model.FLAGS.context_len, model.FLAGS.question_len):

        pred_start_batch, pred_end_batch = model.get_start_end_pos(session, batch)

        pred_start_batch = pred_start_batch.tolist()
        pred_end_batch = pred_end_batch.tolist()

        for ex_idx, (pred_start, pred_end) in enumerate(zip(pred_start_batch, pred_end_batch)):

            context_tokens = batch.context_tokens[ex_idx] 

            assert pred_start in range(len(context_tokens))
            assert pred_end in range(len(context_tokens))

            pred_ans_tokens = context_tokens[pred_start : pred_end +1] 

            uuid = batch.uuids[ex_idx]
            uuid2ans[uuid] = detokenizer.detokenize(pred_ans_tokens, return_str=True)

        batch_num += 1

        if batch_num % 10 == 0:
            print ("Generated answers for %i/%i batches = %.2f%%" % (batch_num, num_batches, batch_num*100.0/num_batches))

    print ("Finished generating answers for dataset.")

    return uuid2ans
