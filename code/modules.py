"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)
            
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)
            fw_out=tf.nn.dropout(fw_out, self.keep_prob)

            return fw_out,out


class Coattention(object):
    """
    Co-attention module
    """

    def __init__(self):
        pass
        
    def build_graph(self, batch_sz, ques, context, question_len, context_len, hidden_size, keep_prob):
        with vs.variable_scope("Coattention"):
            self.hidden_size = hidden_size
            self.keep_prob = keep_prob
            self.context_len= context_len+1
            self.question_len = question_len+1
            self.batch_len= x_shape = context.get_shape().as_list()[0]#batch_sz  #tf.reduce_sum(mask, reduction_indices=1)
            
            #Initialize variables
            theInitializer= tf.contrib.layers.xavier_initializer(dtype = tf.float32)
            fn = lambda x: tf.concat([x, tf.zeros([1, hidden_size], dtype=tf.float32)],0)
            self.D  = tf.map_fn(lambda x: fn(x), context, dtype=tf.float32)
            
            fn1 = lambda x: tf.concat([x, tf.zeros([1, hidden_size], dtype=tf.float32)],0)
            self.ques  = tf.map_fn(lambda x: fn(x), ques, dtype=tf.float32)
            
            self.Wq= tf.get_variable(name = 'Wp', shape = (self.hidden_size, self.hidden_size), initializer = theInitializer, dtype = tf.float32)
            self.b = tf.get_variable(name = 'b_q', shape = (1, self.hidden_size), initializer = theInitializer, dtype = tf.float32)
            self.Wt= tf.get_variable(name = 'Wt', shape = (self.hidden_size, self.hidden_size), initializer = theInitializer, dtype = tf.float32)
            self.rnn_cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
            self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
            self.rnn_cell_bw = tf.contrib.rnn.LSTMCell(hidden_size)
            self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)
    
            #Compute question to context and context to question co-attention
            qs_nw= tf.reshape(self.ques, [-1, self.hidden_size])
            wq_hq= tf.matmul(qs_nw, self.Wq)
            wq_hq = tf.reshape(wq_hq, [-1, self.question_len, self.hidden_size])
            self.Q= tf.tanh(wq_hq+self.b)
            assert self.Q.get_shape().as_list() == [None, self.question_len, self.hidden_size]
            assert self.D.get_shape().as_list() == [None, self.context_len, self.hidden_size]
        
            Q_nw= tf.transpose(self.Q, [0,2,1])
            L= tf.matmul(tf.tensordot(self.D, self.Wt, [[2], [0]]),Q_nw)
            assert L.get_shape().as_list() == [None, self.context_len, self.question_len]
            L_tranpose= tf.transpose(L, [0,2,1])
            A_q= tf.nn.softmax(L_tranpose)
            A_d= tf.nn.softmax(L)
            assert A_d.get_shape().as_list() == [None, self.context_len, self.question_len]
            C_q = tf.matmul(A_q, self.D)
            assert C_q.get_shape().as_list() == [None, self.question_len, self.hidden_size]
            
            
            C_q_q= tf.concat([Q_nw, tf.transpose(C_q, perm=[0, 2 ,1])], 1)
            assert C_q_q.get_shape().as_list() == [None, 2*self.hidden_size, self.question_len]
            assert A_d.get_shape().as_list() == [None, self.context_len, self.question_len]
            C_d = tf.matmul(C_q_q, tf.transpose(A_d, perm=[0, 2 ,1]))
            assert C_d.get_shape().as_list() == [None, 2*self.hidden_size, self.context_len]
            coattention = tf.concat([self.D, tf.transpose(C_d, perm=[0, 2, 1])],2)
            assert coattention.get_shape().as_list() == [None, self.context_len, 3*self.hidden_size]
            co_att, _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, coattention,dtype=tf.float32)
            self.co_attc = tf.concat( co_att,2)
            
            return self.co_attc
            
class MatchLSTM(tf.nn.rnn_cell.BasicLSTMCell):
    """
    Extension of LSTM cell to do match-LSTM
    """
    def __init__(self, hidden_size, ques, context_len, question_len):
        
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
        
        hr = state[1]
        hp_i = inputs

        wp_hp = tf.matmul(hp_i,self.Wp) + tf.matmul(hr, self.Wr) + self.bp
        wp_hp = tf.tile(tf.expand_dims(wp_hp,1),[1,self.question_len,1])
        
        # Yeah pretty sure we need this lol
        G_i = tf.tanh(self.wq_hq + wp_hp)

        # Reshape to multiply against w
        G_i_nw = tf.reshape(G_i, [-1, self.hidden_size])
        a_i = tf.matmul(G_i_nw, self.W) + self.b
        a_i = tf.reshape(a_i, [-1, self.question_len, 1])

        # Prepare dims, and mult attn with question representation in each element of the batch
        ques_nw = tf.transpose(self.ques, [0,2,1])
        z_comp = tf.matmul(ques_nw, a_i)
        z_comp = tf.squeeze(z_comp, [2])

        z_i = tf.concat([hp_i, z_comp],1)

        # Return resultant hr and state from super class (BasicLSTM) run with z_i as input and current state given to our cell
        hr, state = super(MatchLSTM, self).__call__(z_i, state)

        return hr, state

class MatchLSTMDecoder(object):
    """
    Decoder for match-LSTM
    """
    def __init__(self, hidden_size, context_len, question_len):
        self.hidden_size= hidden_size
        self.context_len= context_len
        self.question_len= question_len
        theInitializer= tf.uniform_unit_scaling_initializer(1.0)
        self.V = tf.get_variable(name="V", shape=(2*self.hidden_size,self.hidden_size), initializer=theInitializer)   
        self.Wa = tf.get_variable(name="Wa", shape=(self.hidden_size,self.hidden_size), initializer=theInitializer)
        self.ba = tf.Variable(tf.zeros([1,self.hidden_size]), name = "ba")
        self.vt = tf.Variable(tf.zeros([self.hidden_size,1]), name = "vt")
        self.c = tf.Variable(tf.zeros([1]), name = "c")
        
        
    def decode(self, blended_reps, context_mask, hk):
        mask = tf.log(tf.cast(context_mask, tf.float32))
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
            assert st.get_shape().as_list() == [None, self.context_len, self.hidden_size] 
            
            Fk = tf.tanh(ft + st)
            
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
    Module to take set of hidden states and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks,flag):
        with vs.variable_scope("SimpleSoftmaxLayer"):
            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None)
            logits = tf.squeeze(logits, axis=[2])

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1,flag)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) 
            attn_logits = tf.matmul(keys, values_t) 
            attn_logits_mask = tf.expand_dims(values_mask, 1) 
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2, False)

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) 

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim,flag):
    """
    Takes masked softmax over given dimension of logits.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) 
    if flag==True:
        logits1= logits[:,:-1]
    else:
        logits1=logits
        
    masked_logits = tf.add(logits1, exp_mask) 
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
