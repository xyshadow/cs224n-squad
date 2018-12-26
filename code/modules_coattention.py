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
from modules import masked_softmax
from modules import RNNEncoder


class CoAttn(object):
    """Module BiDAF attention
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

    def build_graph(self, values, values_mask, keys, keys_mask):
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
        with tf.variable_scope("coattention"):
            
            ## affinity matrix
            with tf.variable_scope('affinity_matrix'):
                
                V_project = tf.layers.dense(values,self.value_vec_size,kernel_initializer=tf.contrib.layers.xavier_initializer(),name='q_project',activation=tf.nn.tanh)
                
                ## add c & q zero vector
                k_zero = tf.get_variable('c_zero', shape=(1, 1, self.key_vec_size,), dtype=tf.float32, initializer=tf.random_uniform_initializer)
                v_zero = tf.get_variable('q_zero', shape=(1, 1, self.value_vec_size,), dtype=tf.float32, initializer=tf.random_uniform_initializer)
                
                ## add zero vector to C & Q_project
                K_zero = tf.concat([keys, tf.tile(k_zero, [tf.shape(keys)[0],1,1])],axis=1)
                V_zero = tf.concat([V_project, tf.tile(v_zero, [tf.shape(V_project)[0],1,1])],axis=1)
                
                ## affinity matrix
                #shape (batch * (n + 1) * (m + 1))
                L = tf.matmul(K_zero, tf.transpose(V_zero,perm=(0,2,1)))

            
            ## cortext to question attension
            with tf.variable_scope('co_attention'):
                
                ## add a 1 to the last added vector
                keys_mask_prime = tf.expand_dims(tf.concat((keys_mask,tf.ones((tf.shape(keys_mask)[0],1),dtype=tf.int32)),axis=-1), -1)
                ## add a 1 to the last added vector
                values_mask_prime = tf.expand_dims(tf.concat((values_mask,tf.ones((tf.shape(values_mask)[0],1),dtype=tf.int32)),axis=-1), -1)

                ## shape (batch * (n + 1) * (m + 1))
                _, k2v_attension_ratio= masked_softmax(L,keys_mask_prime,-1)
                ## shape (batch * (n+1) * h)
                k2v_attension = tf.matmul(k2v_attension_ratio,V_zero)
                ## shape (batch * (m + 1) * (n + 1))
                _, v2k_attension_ratio= masked_softmax(tf.transpose(L,(0,2,1)),values_mask_prime,-1)
                ## shape (batch * (m+1) * h)
                v2k_attension = tf.matmul(v2k_attension_ratio,K_zero)

                ## shape (batch * n * h) second attension
                S = tf.matmul(k2v_attension_ratio,v2k_attension)[:,:-1,:]
            
            ## output
            with tf.variable_scope('output_rnn'):
                
                rnn_input = tf.concat([S,k2v_attension[:,:-1,:]],axis=-1)
                encoder = RNNEncoder(2*self.value_vec_size, self.keep_prob,cell_type='lstm')

                output = encoder.build_graph(rnn_input,None)

                # Apply dropout
                output = tf.nn.dropout(output, self.keep_prob)

            return None, output

