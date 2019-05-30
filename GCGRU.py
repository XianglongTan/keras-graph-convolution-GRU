#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:48:05 2019

@author: xianglongtan
"""

from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.regularizers import l2
import scipy.sparse as sp

# Hyper params
num_nodes = len(roadMap)
adj_mx = roadMap.values
batch_size=1

def calculate_laplacian(adj):
    d = adj.sum(1)
    d = np.diag(d)
    return d+adj

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2.T (D-A) D^-1/2 = I - D^-1.T/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    #d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    #normalized_laplacian = np.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_laplacian

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1),dtype='float')
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.astype('int')

def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max,_ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M,_ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2/lambda_max*L)-I
    return L.astype(np.float32)



class GCRNNCell(Layer):
    
    def __init__(self, units, num_proj=None,adj_mx=adj_mx,filter_type='dual_random_walk',
                 activation=None,
                 use_bias=False,
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
        super(GCRNNCell,self).__init__(**kwargs)
        self.units = units//num_nodes
        self.state_size = units
        self.num_proj = num_proj
        try:
            self.activation = activations.get(activation)
        except:
            self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        #self.support = support
        #assert support >= 1
        self.supports = []
        if filter_type == 'norm_laplacian':
            s = K.variable(calculate_scaled_laplacian(adj_mx))
            s.trainable=False
            self.supports.append(s)
        elif filter_type == 'laplacian':
            s = K.variable(calculate_laplacian(adj_mx))
            s.trainable=False
            self.supports.append(s)
        elif filter_type == 'random_walk':
            s = K.variable(calculate_random_walk_matrix(adj_mx).T)
            s.trainable=False
            self.supports.append(s)
        elif filter_type == 'dual_random_walk':
            s = K.variable(calculate_random_walk_matrix(adj_mx).T)
            s.trainable=False
            self.supports.append(s)
            s = K.variable(calculate_reverse_random_walk_matrix(adj_mx).T)
            s.trainable=False
            self.supports.append(s)
        else:
            s = K.variable(calculate_scaled_laplacian(adj_mx))
            s.trainable=False
            self.supports.append(s)
        self.support = len(self.supports)
        
    def build(self, input_shape):
        features_shape = input_shape
        assert len(features_shape) == 2
        input_dim = features_shape[1]//num_nodes
        
        self.r_kernel = self.add_weight(shape=((input_dim+self.units)*self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='r_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.u_kernel = self.add_weight(shape=((input_dim+self.units)*self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='u_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.c_kernel = self.add_weight(shape=((input_dim+self.units)*self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='c_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.num_proj is not None:
            self.proj_kernel = self.add_weight(shape=(self.units,
                                                 self.num_proj),
                                          initializer=self.kernel_initializer,
                                          name='c_kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.r_bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='r_bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.u_bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='u_bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.c_bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='c_bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.proj_bias = self.add_weight(shape=(self.num_proj,),
                                        initializer=self.bias_initializer,
                                        name='c_bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.r_bias = None
            self.u_bias = None
            self.c_bias = None
            self.proj_bias = None
        self.built = True
     
    
    def _gconv(self, h, state, kernel, bias, activation):
        # h: [bs, nodes, inp_dim*support]
        # s: [bs, nodes, units*support]
        inp_and_state = K.concatenate([h,state],axis=-1) 
        x = inp_and_state #[bs, nodes, (units+inp_dim)*supports]
        x = K.dot(x, kernel)
        if self.use_bias:
            x = K.bias_add(x, bias)
        return activation(x)
    
    def call(self, inputs, states):
        batch_size = K.int_shape(inputs)[0] # bs 
        inp_dim = K.int_shape(inputs)[1] # num_nodes*inp_dim
        prev_output = states[0] # states=[bs,nodes*units]
        features = inputs # inputs:[bs, nodes*inp_dim]
        if batch_size is not None:
            features = K.reshape(features,(batch_size,inp_dim//num_nodes,num_nodes)) # inputs:[bs, inp_dim, nodes]
            prev_output = K.reshape(prev_output,(batch_size,self.units,num_nodes)) # states:[bs, units, nodes]
        else:
            features = K.reshape(features,(-1,inp_dim//num_nodes,num_nodes)) # inputs:[bs, inp_dim, nodes]
            prev_output = K.reshape(prev_output,(-1,self.units,num_nodes)) # states:[bs, units, nodes]    
        results_inp = []
        results_states = []
        
        # graph convolution
        for support in self.supports:
            results_inp.append(K.dot(features,K.cast(K.transpose(K.to_dense(support)),'float32')))
            results_states.append(K.dot(prev_output, K.cast(K.transpose(K.to_dense(support)), 'float32')))
        h = K.concatenate(results_inp,axis=1)
        state = K.concatenate(results_states,axis=1)
        h = K.permute_dimensions(h,(0,2,1))
        state = K.permute_dimensions(state,(0,2,1))
        # gated recurrent
        r = self._gconv(h,state,self.r_kernel,self.r_bias,activations.get('sigmoid'))
        u = self._gconv(h,state,self.u_kernel,self.u_bias,activations.get('sigmoid'))
        prev_output = K.permute_dimensions(prev_output,(0,2,1))
        r_prev = K.permute_dimensions(r*prev_output,(0,2,1))
        c_states = []
        for support in self.supports:
            c_states.append(K.dot(r_prev, K.cast(K.transpose(K.to_dense(support)), 'float32')))
        c_state = K.concatenate(c_states,axis=1)
        c_state = K.permute_dimensions(c_state,(0,2,1))
        c = self._gconv(h,c_state,self.c_kernel,self.c_bias,activations.get('tanh'))
        z = u*prev_output+(1-u)*c
        output = self.activation(z)
        output = K.reshape(output, (-1, self.units*num_nodes))
        return output, [output]
    
    
    
def create_GCRNN_model(units = 8, inputs_dim=2, num_nodes=num_nodes, batch_size=batch_size): 
    cell = GCRNNCell(units*num_nodes,use_bias=False, activation= 'relu',filter_type='norm_laplacian')
    cell_out = GCRNNCell(2*num_nodes,use_bias=False, activation= 'relu',filter_type='norm_laplacian')
    inp = Input((None, inputs_dim*num_nodes))
    gcn_in = inp
    gcn_in = RNN(cell,return_sequences=True)(gcn_in)
    gcn_in = Dropout(0.25)(gcn_in)
    #gcn_in = RNN(cell,return_sequences=True)(gcn_in)
    layer_out = RNN(cell_out)
    gcn_out = layer_out(gcn_in)
    model = Model(inp, gcn_out)
    return model

gcrnn = create_GCRNN_model(units=16)
gcrnn.compile(loss='mean_absolute_error', optimizer=Adam(1e-4))
gcrnn.summary()
