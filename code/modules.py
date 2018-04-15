# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)
            
            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)
            fw_out=tf.nn.dropout(fw_out, self.keep_prob)

            return fw_out,out


class Coattention(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        pass
        
    def build_graph(self, batch_sz, ques, context, question_len, context_len, hidden_size, keep_prob):
        with vs.variable_scope("Coattention"):
            self.hidden_size = hidden_size
            self.keep_prob = keep_prob
            self.context_len= context_len+1
            self.question_len = question_len+1
            self.batch_len= x_shape = context.get_shape().as_list()[0]#batch_sz  #tf.reduce_sum(mask, reduction_indices=1)
            
            
            theInitializer= tf.uniform_unit_scaling_initializer(1.0)
            #self.context_sentinel = tf.get_variable(name = 'c_sent', shape = (self.batch_len, 1, self.hidden_size), initializer = theInitializer, dtype = tf.float32)
            #self.ques_sentinel = tf.get_variable(name = 'q_sent', shape = (self.batch_len, 1, self.hidden_size), initializer = theInitializer, dtype = tf.float32)
            
            fn = lambda x: tf.concat([x, tf.zeros([1, hidden_size], dtype=tf.float32)],0)
            self.D  = tf.map_fn(lambda x: fn(x), context, dtype=tf.float32)
            
            fn1 = lambda x: tf.concat([x, tf.zeros([1, hidden_size], dtype=tf.float32)],0)
            self.ques  = tf.map_fn(lambda x: fn(x), ques, dtype=tf.float32)
            
            #self.ques= tf.concat([ques, self.ques_sentinel], 1)
            #self.D = tf.concat([context, self.context_sentinel], 1)
            self.Wq= tf.get_variable(name = 'Wp', shape = (self.hidden_size, self.hidden_size), initializer = theInitializer, dtype = tf.float32)
            self.b = tf.get_variable(name = 'b_q', shape = (1, self.hidden_size), initializer = theInitializer, dtype = tf.float32)
            self.rnn_cell_fw = tf.contrib.rnn.LSTMCell(hidden_size) #rnn_cell.GRUCell(self.hidden_size)
            self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
            self.rnn_cell_bw = tf.contrib.rnn.LSTMCell(hidden_size) #rnn_cell.GRUCell(self.hidden_size)
            self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)
    
            
            qs_nw= tf.reshape(self.ques, [-1, self.hidden_size])
            wq_hq= tf.matmul(qs_nw, self.Wq)
            wq_hq = tf.reshape(wq_hq, [-1, self.question_len, self.hidden_size])
            self.Q= tf.tanh(wq_hq+self.b)
            assert self.Q.get_shape().as_list() == [None, self.question_len, self.hidden_size]
            assert self.D.get_shape().as_list() == [None, self.context_len, self.hidden_size]
        
            Q_nw= tf.transpose(self.Q, [0,2,1])
            L= tf.matmul(self.D, Q_nw)
            #print(self.D.get_shape().as_list())
            assert L.get_shape().as_list() == [None, self.context_len, self.question_len]
            L_tranpose= tf.transpose(L, [0,2,1])
            A_q= tf.nn.softmax(L_tranpose)
            A_d= tf.nn.softmax(L)
            assert A_d.get_shape().as_list() == [None, self.context_len, self.question_len]
            #assert sef.D.get_shape().as_list() == [None, self.context_len, self.question_len]
            C_q = tf.matmul(A_q, self.D)
            assert C_q.get_shape().as_list() == [None, self.question_len, self.hidden_size]
            
            
            C_q_q= tf.concat([Q_nw, tf.transpose(C_q, perm=[0, 2 ,1])], 1)
            assert C_q_q.get_shape().as_list() == [None, 2*self.hidden_size, self.question_len]
            assert A_d.get_shape().as_list() == [None, self.context_len, self.question_len]
            C_d = tf.matmul(C_q_q, tf.transpose(A_d, perm=[0, 2 ,1]))
            assert C_d.get_shape().as_list() == [None, 2*self.hidden_size, self.context_len]
            # final coattention context, (batch_size, context+1, 3*hidden_size)
            #assert self.D.get_shape().as_list() == [None, 2*self.hidden_size, self.context_len]
            coattention = tf.concat([self.D, tf.transpose(C_d, perm=[0, 2, 1])],2)
            assert coattention.get_shape().as_list() == [None, self.context_len, 3*self.hidden_size]
            co_att, _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, coattention,dtype=tf.float32)
            self.co_attc = tf.concat( co_att,2)
            #assert co_att.get_shape().as_list() == [None, self.context_len, 3*self.hidden_size]
            
            return self.co_attc

class MatchLSTM(tf.nn.rnn_cell.BasicLSTMCell):
    """
    Extension of LSTM cell to do matching and magic. Designed to be fed to dynammic_rnn
    """
    def __init__(self, hidden_size, ques, context_len, question_len):
        # Uniform distribution, as opposed to xavier, which is normal
        #self.HQ = HQ
        #self.hidden_size = hidden_size
        #self.context_len = context_len
        #self.question_len=question_len

        #l, P, Q = self.hidden_size, self.FLAGS.max_paragraph_size, self.FLAGS.max_question_size
        #self.WQ = tf.get_variable("WQ", [self.hidden_size,self.hidden_size], initializer=tf.uniform_unit_scaling_initializer(1.0)) 
        #self.WP = tf.get_variable("WP", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))
        #self.WR = tf.get_variable("WR", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))

        #self.bP = tf.Variable(tf.zeros([1, l]))
        #self.w = tf.Variable(tf.zeros([l,1])) 
        #self.b = tf.Variable(tf.zeros([1,1]))

        # Calculate term1 by resphapeing to l
        #HQ_shaped = tf.reshape(HQ, [-1, l])
        #term1 = tf.matmul(HQ_shaped, self.WQ)
        #term1 = tf.reshape(term1, [-1, Q, l])
        #self.term1 = term1
        
        self.ques=ques
        self.context_len= context_len
        self.question_len = question_len
        self.hidden_size = hidden_size
        
        theInitializer= tf.uniform_unit_scaling_initializer(1.0)
        self.Wp = tf.get_variable(name = 'Wp', shape = (self.hidden_size, self.hidden_size), initializer = theInitializer, dtype = tf.float32)
        self.Wq = tf.get_variable(name = 'Wq', shape = (self.hidden_size, self.hidden_size), initializer = theInitializer, dtype = tf.float32)
        self.Wr = tf.get_variable(name = 'Wr', shape = (self.hidden_size, self.hidden_size), initializer = theInitializer, dtype = tf.float32)
        
        self.bp = tf.Variable(tf.zeros([1, self.hidden_size]))
        self.W = tf.Variable(tf.zeros([self.hidden_size, 1]))
        self.b = tf.Variable(tf.zeros([1, 1]))
        self.context_len= context_len
        self.question_len = question_len
        
        qs_nw= tf.reshape(self.ques, [-1, self.hidden_size])
        self.wq_hq= tf.matmul(qs_nw, self.Wq)
        self.wq_hq = tf.reshape(self.wq_hq, [-1, self.question_len, self.hidden_size])

        super(MatchLSTM, self).__init__(hidden_size)

    def __call__(self, inputs, state, scope = None):
        """
        inputs: a batch representation (HP at each word i) that is inputs = hp_i and are [None, l]
        state: a current state for our cell which is LSTM so its a tuple of (c_mem, h_state), both are [None, l]
        """
        
        #For naming convention load in from self the params and rename
        #term1 = self.term1
        #WQ, WP, WR = self.WQ, self.WP, self.WR
        #bP, w, b = self.bP, self.w, self.b
        #l, P, Q = self.hidden_size, self.FLAGS.max_paragraph_size, self.FLAGS.max_question_size
        #ques = self.HQ
        hr = state[1]
        hp_i = inputs

        # Check correct input dimensions
        #assert hr.get_shape().as_list() == [None, l]
        #assert hp_i.get_shape().as_list() == [None, l]

        # Way to extent a [None, l] matrix by dim Q 
        wp_hp = tf.matmul(hp_i,self.Wp) + tf.matmul(hr, self.Wr) + self.bp
        wp_hp = tf.tile(tf.expand_dims(wp_hp,1),[1,self.question_len,1])
        #term2 = tf.transpose(tf.stack([term2 for _ in range(Q)]), [1,0,2])

        # Check correct term dimensions for use
        #assert term1.get_shape().as_list() == [None, Q, l]
        #assert term2.get_shape().as_list() == [None, Q, l]

        # Yeah pretty sure we need this lol
        G_i = tf.tanh(self.wq_hq + wp_hp)

        # Reshape to multiply against w
        G_i_nw = tf.reshape(G_i, [-1, self.hidden_size])
        a_i = tf.matmul(G_i_nw, self.W) + self.b
        a_i = tf.reshape(a_i, [-1, self.question_len, 1])

        # Check that the attention matrix is properly shaped (3rd dim useful for batch_matmul in next step)
        #assert a_i.get_shape().as_list() == [None, Q, 1]

        # Prepare dims, and mult attn with question representation in each element of the batch
        ques_nw = tf.transpose(self.ques, [0,2,1])
        z_comp = tf.matmul(ques_nw, a_i)
        z_comp = tf.squeeze(z_comp, [2])

        # Check dims of above operation
        #assert z_comp.get_shape().as_list() == [None, l]

        # Concatenate elements for feed into LSTM
        z_i = tf.concat([hp_i, z_comp],1)

        # Check dims of LSTM input
        #assert z_i.get_shape().as_list() == [None, 2*l]

        # Return resultant hr and state from super class (BasicLSTM) run with z_i as input and current state given to our cell
        hr, state = super(MatchLSTM, self).__call__(z_i, state)

        return hr, state

class MatchLSTMDecoder(object):
    def __init__(self, hidden_size, context_len, question_len):
        self.hidden_size= hidden_size
        self.context_len= context_len
        self.question_len= question_len
        #self.state = tf.zeros([1, self.hidden_size])
        theInitializer= tf.uniform_unit_scaling_initializer(1.0)
        self.V = tf.get_variable(name="V", shape=(2*self.hidden_size,self.hidden_size), initializer=theInitializer)   
        self.Wa = tf.get_variable(name="Wa", shape=(self.hidden_size,self.hidden_size), initializer=theInitializer)
        self.ba = tf.Variable(tf.zeros([1,self.hidden_size]), name = "ba")
        self.vt = tf.Variable(tf.zeros([self.hidden_size,1]), name = "vt")
        self.c = tf.Variable(tf.zeros([1]), name = "c")
        
        
    def decode(self, blended_reps, context_mask, hk):
        mask = tf.log(tf.cast(context_mask, tf.float32))
        #hk= (state,state)
        st_nw=(hk,hk)
        indx=[None, None]
        for i in range(0,2):
            if i != 0:
                tf.get_variable_scope().reuse_variables()
                
            hr=tf.reshape(blended_reps,[-1,2*self.hidden_size])
            ft= tf.matmul(hr , self.V)
            ft = tf.reshape(ft, [-1, self.context_len, self.hidden_size])
            assert ft.get_shape().as_list() == [None, self.context_len, self.hidden_size] 
            
            assert hk.get_shape().as_list() == [None, self.hidden_size]
            st= tf.matmul(hk,self.Wa)+self.ba
            st = tf.tile(tf.expand_dims(st,1),[1,self.context_len,1])
            #term2 = tf.transpose(tf.stack([term2 for _ in range(P)]), [1,0,2]) 
            assert st.get_shape().as_list() == [None, self.context_len, self.hidden_size] 
            
            Fk = tf.tanh(ft + st)
            #assert Fk.get_shape().as_list() == [None, P, l] 
    
            # Generate beta_term v^T * Fk + c * e(P)
            Fk_nw = tf.reshape(Fk, [-1, self.hidden_size])
            bk = tf.matmul(Fk_nw, self.vt) + self.c
            bk= tf.reshape(bk ,[-1, self.context_len, 1])
            assert bk.get_shape().as_list() == [None, self.context_len, 1] 
    
            #TEST OTHER MASK VERSION
            bk_ms = tf.squeeze(bk,2) + mask
            assert bk_ms.get_shape().as_list() == [None, self.context_len] 
    
            # Get Beta (prob dist over the paragraph)
            bk_nw = tf.nn.softmax(bk_ms)
            bk_nw = tf.expand_dims(bk_nw, 2)
            assert bk_nw.get_shape().as_list() == [None, self.context_len, 1]  
    
            # Setup input to LSTM
            Hr_nw = tf.transpose(blended_reps, [0, 2, 1])
            hr_in = tf.squeeze(tf.matmul(Hr_nw, bk_nw), [2])
            assert hr_in.get_shape().as_list() == [None, 2*self.hidden_size] 
    
            # Ouput and State for next iteration
            lstm=tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            hk, st_nw = lstm(hr_in, st_nw)
    
            #Save a 2D rep of Beta as output
            indx[i] = bk_ms 

        return tuple(indx)
        
        
class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks,flag):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1,flag)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2, False) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim,flag):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    if flag==True:
        logits1= logits[:,:-1]
    else:
        logits1=logits
        
    masked_logits = tf.add(logits1, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
