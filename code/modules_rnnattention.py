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
from modules import RNNEncoder, BasicAttn



class RNNAttn(object):
    """Module BiDAF attention
    """

    def __init__(self, keep_prob, value_vec_size, key_vec_size):
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
        Take the attention output of the previous attension layers, and add self attention
        using multi-head attention
        """
        with tf.variable_scope("rnnattention"):
            ## one layer of normal attention
            attn_layer = BasicAttn(self.keep_prob, self.value_vec_size, self.key_vec_size)
            _, attn_output = attn_layer.build_graph(values, values_mask, keys, keys_mask)
            
            ## concat attn output and keys
            key_attn = tf.concat([keys, attn_output], axis=2)
            key_attn = tf.contrib.layers.fully_connected(key_attn, num_outputs=self.key_vec_size/2)
            
            ## add a layer of rnn
            encoder = RNNEncoder(self.key_vec_size/2, self.keep_prob, cell_type='lstm')
            
            rnn_output = encoder.build_graph(key_attn,None)
            
            ## self attension, transformer stype, shape of self_attns is (batch,length,hidden)
            return None, rnn_output
            
        

    