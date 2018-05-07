"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn, MatchLSTM, MatchLSTMDecoder, Coattention

logging.basicConfig(level=logging.INFO)


class QAModel(object):
    """Top-level Question Answering module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        """
        Initializes the QA model.
        """
        print ("Initializing the QAModel...")
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        self.FLAGS.deep= True
        # Add all parts of the graph
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.xavier_initializer(uniform = True, dtype = tf.float32)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph_coattention()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.decayed_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step, decay_steps = 1000, decay_rate = 0.88, staircase=True)
        opt = tf.train.AdamOptimizer(self.decayed_rate)
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()

    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        self.context_ids = tf.placeholder(tf.int32, (None, self.FLAGS.context_len), name="aa")
        self.context_mask = tf.placeholder(tf.int32, (None, self.FLAGS.context_len),name="bb")
        self.qn_ids = tf.placeholder(tf.int32, (None, self.FLAGS.question_len),name="cc")
        self.qn_mask = tf.placeholder(tf.int32, (None, self.FLAGS.question_len),name="dd")
        self.ans_span = tf.placeholder(tf.int32, (None, 2),name="ee")
        self.context_length = tf.placeholder(tf.int32, (None), name="context_length")
        self.state = tf.placeholder(tf.float32, (None, self.FLAGS.hidden_size), name="state")
        self.keep_prob = tf.placeholder_with_default(1.0, (),name="rr")


    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.
        """
        with vs.variable_scope("embeddings"):

            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)


    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.
        """

        # Use a RNN to get hidden states for the context and the question
        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        _,context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
        _,question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # Use context hidden states to attend to question hidden states
        attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens) # attn_output is shape (batch_size, context_len, hidden_size*2)

        # Concat attn_output to context_hiddens to get blended_reps
        blended_reps = tf.concat([context_hiddens, attn_output], axis=2) # (batch_size, context_len, hidden_size*4)

        # Apply fully connected layer to each blended representation
        blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size) # blended_reps_final is shape (batch_size, context_len, hidden_size)

        # Use softmax layer to compute probability distribution for start location
        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask, False)

        # Use softmax layer to compute probability distribution for end location
        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask, False)

    
    
    
    def build_graph_coattention(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.
        """

        # Use a RNN to get hidden states for the context and the question
        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        _,context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
        _,question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # Compute both sided attention
        coatt= Coattention()
        co_att= coatt.build_graph(self.FLAGS.batch_size,question_hiddens, context_hiddens, self.FLAGS.question_len, self.FLAGS.context_len, 2*self.FLAGS.hidden_size, self.keep_prob)
        
        co_att_final = tf.contrib.layers.fully_connected(co_att, num_outputs=self.FLAGS.hidden_size)
        # Use softmax layer to compute probability distribution for start location
        with vs.variable_scope("StartDist") as scp:
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(co_att_final, self.context_mask, True)
            scp.reuse_variables()
        # Use softmax layer to compute probability distribution for end location
        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(co_att_final, self.context_mask, True)
    
    
    
    def build_graph_match(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.
        """

        # Use a RNN to get hidden states for the context and the question
        context_hd = tf.nn.dropout(self.context_embs, 0.6)
        question_hd = tf.nn.dropout(self.qn_embs, 0.6)

        #Preprocessing LSTM
        with tf.variable_scope("question_encode"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.hidden_size) #self.size passed in through initialization from "state_size" flag
            _,context_hiddens = tf.nn.dynamic_rnn(cell, context_hd, dtype = tf.float32)

        with tf.variable_scope("paragraph_encode"):
            cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.hidden_size)
            _,question_hiddens = tf.nn.dynamic_rnn(cell2, question_hd, dtype = tf.float32)   #sequence length masks dynamic_rnn

        
        with tf.variable_scope("forward"):
            forward = MatchLSTM(self.FLAGS.hidden_size, question_hiddens, self.FLAGS.context_len,self.FLAGS.question_len) 

            if (self.FLAGS.deep):
                forward = [forward] + [tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.hidden_size)]*2
                forward = tf.nn.rnn_cell.MultiRNNCell(forward)

        with tf.variable_scope("backward"):
            backward = MatchLSTM(self.FLAGS.hidden_size, question_hiddens, self.FLAGS.context_len,self.FLAGS.question_len) 

            if (self.FLAGS.deep):
                backward = [backward] + [tf.nn.rnn_cell.BasicLSTMCell(self.FLAGS.hidden_size)]*2
                backward = tf.nn.rnn_cell.MultiRNNCell(backward)
        # Calculate encodings for both forward and backward directions
        (right, left), _ = tf.nn.bidirectional_dynamic_rnn(forward, backward, context_hiddens, dtype = tf.float32)
        
        blended_reps = tf.concat([right, left],2)
        
        dr = (self.keep_prob)/2.0 #+ (1-self.keep_prob)
        blended_reps = tf.nn.dropout(blended_reps, dr)
        assert blended_reps.get_shape().as_list() == [None, self.FLAGS.context_len, 2*self.FLAGS.hidden_size]   
        dec= MatchLSTMDecoder(self.FLAGS.hidden_size, self.FLAGS.context_len,self.FLAGS.question_len)
        indxs_st, indxs_en= dec.decode(blended_reps, self.context_mask,self.state)
        self.logits_end= indxs_en
        self.logits_start=indxs_st
        self.probdist_start = tf.nn.softmax(self.logits_start)
        self.probdist_end = tf.nn.softmax(self.logits_end)
        
    def add_loss(self):
        """
        Add loss computation to the graph.
        """
        with vs.variable_scope("loss"):
            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Add the two losses
            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)
            

    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = list(batch.context_ids)
        input_feed[self.context_mask] = list(batch.context_mask)
        input_feed[self.qn_ids] = list(batch.qn_ids)
        input_feed[self.qn_mask] = list(batch.qn_mask)
        input_feed[self.ans_span] = list(batch.ans_span)
        input_feed[self.keep_prob] = 1-self.FLAGS.dropout # apply dropout
        input_feed[self.context_length]= np.sum(list(batch.context_mask), axis = 1)
        input_feed[self.state] = np.zeros((len(batch.qn_ids), self.FLAGS.hidden_size))

        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]
        
        # Run the model
        _, summaries, loss, global_step, param_norm, gradient_norm = session.run(output_feed, feed_dict= input_feed)
        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.
        """

        input_feed = {}
        input_feed[self.context_ids] = list(batch.context_ids)
        input_feed[self.context_mask] = list(batch.context_mask)
        input_feed[self.qn_ids] = list(batch.qn_ids)
        input_feed[self.qn_mask] = list(batch.qn_mask)
        input_feed[self.ans_span] = list(batch.ans_span)
        input_feed[self.context_length]= np.sum(list(batch.context_mask), axis = 1)
        input_feed[self.state] = np.zeros((len(batch.qn_ids), self.FLAGS.hidden_size))
        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.
        """
        input_feed = {}
        input_feed[self.context_ids] = list(batch.context_ids)
        input_feed[self.context_mask] = list(batch.context_mask)
        input_feed[self.qn_ids] = list(batch.qn_ids)
        input_feed[self.qn_mask] = list(batch.qn_mask)
        input_feed[self.context_length]= np.sum(list(batch.context_mask), axis = 1)
        input_feed[self.state] = np.zeros((len(batch.qn_ids), self.FLAGS.hidden_size))
        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch):
        """
        Run forward-pass only; get the most likely answer span.
        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist = self.get_prob_dists(session, batch)
        
        #Get answer within length 15
        window_size = 15 
        a_s_batch = []
        a_e_batch = []
        for b_s, b_e in zip(start_dist, end_dist):
            a_s, a_e, max_p = 0, 0, 0
            num_elem = len(b_s)
            for start_ind in range(num_elem):
                for end_ind in range(start_ind, min(window_size + start_ind, num_elem)):
                    if(b_s[start_ind]*b_e[end_ind] > max_p):
                        max_p = b_s[start_ind]*b_e[end_ind]
                        a_s = start_ind
                        a_e = end_ind
            a_s_batch.append(a_s)
            a_e_batch.append(a_e)

        return a_s_batch, a_e_batch

    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.
        """
        
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print ("Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic))

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False):

            pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch)

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        exp_loss = None

        # Checkpoint management.
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                if global_step % self.FLAGS.eval_every == 0:

                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    if best_dev_f1 is None or dev_f1 > best_dev_f1:
                        best_dev_f1 = dev_f1
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
