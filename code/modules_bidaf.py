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



class BiDAFAttn(object):
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

    def build_graph(self, values, values_mask, keys, keys_mask=None):
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
        with tf.variable_scope("BiDAFAttn"):
            
            ## similarity vector, trainable variable
            with tf.variable_scope('similarity_matrix'):
                W_sim = tf.get_variable('W_sim', shape=(1, 1, self.key_vec_size,), dtype=tf.float32, initializer=tf.random_normal_initializer)
                S = tf.matmul(keys * W_sim, tf.transpose(values,perm=(0,2,1)))
           
            
            ## cortext to question attension
            with tf.variable_scope('c2q_attention'):
                ## take softmax on S along M, effectively "mask out" the null after real questions
                _, attn_dist = masked_softmax(S, tf.expand_dims(values_mask,1), -1)

                ## weighted sum of Q over each row of A.
                A = tf.matmul(attn_dist,values)
                
            ## question to cortext attension
            with tf.variable_scope('q2c_attention'):
                ## batch X n
                M = tf.reduce_max(S,axis=-1)
                theta = tf.nn.softmax(M, -1)
                ## batch X n
                keys_prime = tf.matmul(tf.expand_dims(theta,1),keys)
            
            ## output
            with tf.variable_scope('gen_output'):
                output = tf.concat([A, keys*A, keys*keys_prime],axis=-1)


                # Apply dropout
                output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

