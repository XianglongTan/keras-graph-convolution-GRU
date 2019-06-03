#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 11:14:46 2019

@author: xianglongtan
"""

from keras import activations,initializers,constraints,regularizers
from keras.engine import Layer
import keras.backend as K

class GraphConvolution(Layer):
    def __init__(self, units, support=1,
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        self.support = support
        assert support >= 1
        
    def compute_output_shape(self, input_shapes):
        features_shape=input_shapes[0] # (batch_size, input_feat)
        output_shape = (features_shape[0], self.units) # (batch_size, output_feat)
        return output_shape # (batch_size, output_dim)
    
    def build(self, input_shapes):
        features_shape = input_shapes[0] # (batch_size, input_feat)
        assert len(features_shape) == 2
        input_dim = features_shape[1] # input_feat

        self.kernel = self.add_weight(shape=(input_dim*self.support, 
                                            self.units), # (input_feat*support, units)
                                     initializer = self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                       initializer = self.bias_initializer,
                                       name='bias',
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True
        
    def call(self, inputs, mask=None):
        # inputs:[X(batch_size, features), ]
        features = inputs[0]
        basis = inputs[1:]
        
        supports = list()
        for i in range(self.support):
            supports.append(K.dot(basis[i], features))
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)
        
        if self.bias:
            output += self.bias
        return self.activation(output)
    
    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items())+list(config.items()))
        
    
from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import *
from keras.engine import Layer

class GraphAttention(Layer):
    
    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
            
        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)
        
        
    def build(self, input_shape):
        assert len(input_shape) >= 2 # at least a feature and an adjcency
        F =  input_shape[0][-1]
        
        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)
            
            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)
                
            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True
        

        
    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)
        
        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head] # W in the paper (F x F')
            attention_kernel = self.attn_kernels[head] # Attention kernel a in the paper (2F' x 1)
            
            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F')
            
            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])   # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1]) # (N x 1), [a_2]^T [Wh_j]
            
            # Attention head a(Wh_i, Wh_j) = a^T[[Wh_i], [Wh_j]]
            dense = attn_for_self + K.transpose(attn_for_neighs) # (N x N) via broadcasting
            
            # Add nonlinearty
            dense = activations.relu(dense,alpha=0.2)
            
            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9*(1.0-A)
            dense += mask
            
            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (N x N)
            
            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')
            
            # Linear combination with neighbors' features
            node_features = K.dot(dropout_attn, dropout_feat) # (N x F')
            
            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])
            
            # Add output of attention head to final output
            outputs.append(node_features)
            
        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = K.concatenate(outputs) # (N x KF')
        else:
            output = K.mean(K.stack(outputs), axis=0) # (N x F')
            
        output = self.activation(output)
        return output
    
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], self.output_dim)
        return output_shape
    