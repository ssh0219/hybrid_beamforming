
# coding: utf-8

# In[46]:


import tensorflow as tf
import numpy as np
import os
import math
import scipy.io as sio
#from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph() 


# In[47]:


L, M, N, K, snr = 5, 10, 1, 10, 5
layer, batch_size, I_max = 3, 20, 200
p_ = tf.random_normal(mean=0.0,stddev=tf.sqrt(0.5),shape=[2*K*M, N])
p = tf.complex(p_[0:K*M,:], p_[K*M:,:])
w_ = tf.random_normal(mean=0.0,stddev=tf.sqrt(0.5),shape=[2*K*L, N])
w = tf.complex(w_[0:K*L,:], w_[K*L:,:])
n_ = tf.random_normal(mean=0.0,stddev=tf.sqrt(tf.sqrt(0.1)*0.5),shape=[2*L, 1])
n = tf.complex(n_[0:L,:], n_[L:,:])
b = tf.placeholder(dtype=tf.complex64, shape=[K,batch_size])
H_ = tf.placeholder(dtype=tf.float32, shape=[2*K*L, M])
H = tf.complex(H_[0:K*L,:], H_[K*L:,:])
lr_p = tf.complex(tf.random_normal(mean=0.0,stddev=tf.sqrt(0.1),shape=[K*M, N]), tf.zeros(shape=[K*M,N]))
lr_w = tf.complex(tf.random_normal(mean=0.0,stddev=tf.sqrt(0.1),shape=[K*L, N]), tf.zeros(shape=[K*L,N]))

error = []
train_b = tf.sign(tf.random_uniform(shape=[10000, K], minval=-1,maxval=1, dtype=tf.float32))
test_b = tf.sign(tf.random_uniform(shape=[3000, K], minval=-1,maxval=1, dtype=tf.float32))


# In[48]:


print(b)
sum2 = tf.complex(tf.zeros(shape=[M,M]), tf.zeros(shape=[M,M]))
rol = tf.complex(tf.zeros(shape=[K,1]), tf.zeros(shape=[K,1]))
for k in range(0,K):
    sum2 = sum2+tf.matmul(p[k*M:(k+1)*M,:], tf.transpose(tf.conj(p[k*M:(k+1)*M,:])))
for k in range(0,K):
    part0 = rol[0:k,:]
    part1 = tf.sqrt(tf.matmul(tf.matmul(tf.matmul(tf.matmul(tf.transpose(tf.conj(w[k*L:(k+1)*L,:])), H[k*L:(k+1)*L,:]), sum2),
                                        tf.transpose(tf.conj(H[k*L:(k+1)*L,:]))), w[k*L:(k+1)*L,:]))
    part2 = rol[k+1:,:]
    rol = tf.concat([part0,part1,part2], axis=0)


# In[49]:


def forward(p,w,lr_p,lr_w):
    b_hat = tf.complex(tf.zeros(shape=[K,batch_size]), tf.zeros(shape=[K,batch_size]))
    delta_w = tf.complex(tf.zeros(shape=[K*L,batch_size]), tf.zeros(shape=[K*L,batch_size]))
    delta_p = tf.complex(tf.zeros(shape=[K*M,batch_size]), tf.zeros(shape=[K*M,batch_size]))
    delta_p_avr = tf.complex(tf.zeros(shape=[K*M,1]), tf.zeros(shape=[K*M,1]))
    delta_w_avr = tf.complex(tf.zeros(shape=[K*L,1]), tf.zeros(shape=[K*L,1]))
    
    sum1 = tf.complex(tf.zeros(shape=[M,batch_size]), tf.zeros(shape=[M,batch_size]))
    part1 = tf.complex(tf.zeros(shape=[M,1]), tf.zeros(shape=[M,1]))
    for i in range(0,batch_size):
        part0 = sum1[:,0:i]
        part1 = tf.complex(tf.zeros(shape=[M,1]), tf.zeros(shape=[M,1]))
        part2 = sum1[:,(i+1):]
        for k in range(0,K):
            part1 = part1+tf.matmul(p[k*M:(k+1)*M,:], b[k:(k+1),i:(i+1)])
        sum1 = tf.concat([part0,part1,part2], axis=1)
    #print(sum1)
    
    for i in range(0,batch_size):
        for k in range(0,K):
            part0 = b_hat[:k,:]
            part1 = b_hat[k:(k+1),:i]
            part2 = tf.matmul(tf.transpose(tf.conj(w[k*L:(k+1)*L,:])), tf.add(tf.matmul(H[k*L:(k+1)*L,:], sum1[:,i:(i+1)]), n))
            part3 = b_hat[k:(k+1),(i+1):]
            part4 = b_hat[(k+1):,:]
            b_hat = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0)
    
    for i in range(0,batch_size):
        for k in range(0,K):
            part0 = delta_p[:k*M,:]
            part1 = delta_p[k*M:(k+1)*M,:i]
            part2 = -(b[k:(k+1),i:(i+1)]/(2*tf.sqrt(2*3.14*rol[k:(k+1),:]))
                      *tf.exp(-tf.complex(tf.pow(tf.real(b_hat[k:(k+1),i:(i+1)]),2), tf.zeros(shape=[1,1]))/(2*tf.pow(rol[k:(k+1),:],2)))
                      *b[k:(k+1),i:(i+1)]*tf.matmul(tf.transpose(tf.conj(H[k*L:(k+1)*L,:])), w[k*L:(k+1)*L,:]))
            part3 = delta_p[k*M:(k+1)*M,(i+1):]
            part4 = delta_p[(k+1)*M:,:]
            delta_p = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0)
    delta_p_avr = tf.reduce_mean(delta_p, axis=1, keep_dims=True)
    p = p-lr_p*delta_p_avr
    
    sum1 = tf.complex(tf.zeros(shape=[M,batch_size]), tf.zeros(shape=[M,batch_size]))
    part1 = tf.complex(tf.zeros(shape=[M,1]), tf.zeros(shape=[M,1]))
    for i in range(0,batch_size):
        part0 = sum1[:,0:i]
        part1 = tf.complex(tf.zeros(shape=[M,1]), tf.zeros(shape=[M,1]))
        part2 = sum1[:,(i+1):]
        for k in range(0,K):
            part1 = part1+tf.matmul(p[k*M:(k+1)*M,:], b[k:(k+1),i:(i+1)])
        sum1 = tf.concat([part0,part1,part2], axis=1)
    #print(sum1)
    
    for i in range(0,batch_size):
        for k in range(0,K):
            part0 = b_hat[:k,:]
            part1 = b_hat[k:(k+1),:i]
            part2 = tf.matmul(tf.transpose(tf.conj(w[k*L:(k+1)*L,:])), tf.add(tf.matmul(H[k*L:(k+1)*L,:], sum1[:,i:(i+1)]), n))
            part3 = b_hat[k:(k+1),(i+1):]
            part4 = b_hat[(k+1):,:]
            b_hat = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0)
    
    for i in range(0,batch_size):
        for k in range(0,K):
            part0 = delta_w[:k*L,:]
            part1 = delta_w[k*L:(k+1)*L,:i]
            part2 = -(b[k:(k+1),i:(i+1)]/(2*tf.sqrt(2*3.14*rol[k:(k+1),:]))
                      *tf.exp(-tf.complex(tf.pow(tf.real(b_hat[k:(k+1),i:(i+1)]),2), tf.zeros(shape=[1,1]))/(2*tf.pow(rol[k:(k+1),:],2)))
                      *tf.matmul(H[k*L:(k+1)*L,:], sum1[:,i:(i+1)]))
            part3 = delta_w[k*L:(k+1)*L,(i+1):]
            part4 = delta_w[(k+1)*L:,:]
            delta_w = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0)
    delta_w_avr = tf.reduce_mean(delta_w, axis=1, keep_dims=True)
    w = w-lr_w*delta_w_avr
    return p,w,delta_p_avr,delta_w_avr


# In[50]:


def back(p,w, G_p,G_w):
    #sum1 = tf.complex(tf.zeros(shape=[M,1]), tf.zeros(shape=[M,1]))
    #b_hat = tf.complex(tf.zeros(shape=[K,1]), tf.zeros(shape=[K,1]))
    #for k in range(0,K):
    #    sum1 = sum1+tf.matmul(p[k*M:(k+1)*M,:], b[k:(k+1),i:(i+1)])
    
    sum1 = tf.complex(tf.zeros(shape=[M,batch_size]), tf.zeros(shape=[M,batch_size]))
    part1 = tf.complex(tf.zeros(shape=[M,1]), tf.zeros(shape=[M,1]))
    for i in range(0,batch_size):
        part0 = sum1[:,0:i]
        part1 = tf.complex(tf.zeros(shape=[M,1]), tf.zeros(shape=[M,1]))
        part2 = sum1[:,(i+1):]
        for k in range(0,K):
            part1 = part1+tf.matmul(p[k*M:(k+1)*M,:], b[k:(k+1),i:(i+1)])
        sum1 = tf.concat([part0,part1,part2], axis=1)
    
    #for k in range(0,K):
    #    part0 = b_hat[:k,:]
    #    part1 = tf.matmul(tf.transpose(tf.conj(w[k*L:(k+1)*L,:])), tf.add(tf.matmul(H[k*L:(k+1)*L,:], sum1), n))
    #    part2 = b_hat[(k+1):,:]
    #    b_hat = tf.concat([part0,part1,part2], axis=0)
    
    b_hat = tf.complex(tf.zeros(shape=[K,batch_size]), tf.zeros(shape=[K,batch_size]))
    for i in range(0,batch_size):
        for k in range(0,K):
            part0 = b_hat[:k,:]
            part1 = b_hat[k:(k+1),:i]
            part2 = tf.matmul(tf.transpose(tf.conj(w[k*L:(k+1)*L,:])), tf.add(tf.matmul(H[k*L:(k+1)*L,:], sum1[:,i:(i+1)]), n))
            part3 = b_hat[k:(k+1),(i+1):]
            part4 = b_hat[(k+1):,:]
            b_hat = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0)
    
    cof1 = tf.complex(tf.zeros(shape=[K,batch_size]), tf.zeros(shape=[K,batch_size]))
    cof2 = tf.complex(tf.zeros(shape=[K,batch_size]), tf.zeros(shape=[K,batch_size]))
    for i in range(0,batch_size):
        part0 = cof1[:,:i]
        part1 = (b[:,i:(i+1)]/(2*tf.sqrt(2*3.14*rol))
                 *tf.exp(-tf.complex(tf.pow(tf.real(b_hat[:,i:(i+1)]),2), tf.zeros(shape=[K,1]))/(2*tf.pow(rol,2))))
        part2 = cof1[:,(i+1):]
        cof1 = tf.concat([part0,part1,part2], axis=1)
        
        part0 = cof2[:,:i]
        part1 = cof1[:,i:(i+1)]*tf.complex(tf.real(b_hat[:,i:(i+1)]), tf.zeros(shape=[K,1]))/tf.pow(rol,2)
        part2 = cof2[:,(i+1):]
        cof2 = tf.concat([part0,part1,part2], axis=1)
    
    
    G_p_index = tf.complex(tf.zeros(shape=[batch_size,K*M]), tf.zeros(shape=[batch_size,K*M]))
    G_w_index = tf.complex(tf.zeros(shape=[batch_size,K*L]), tf.zeros(shape=[batch_size,K*L]))
    for i in range(0,batch_size):
        for k in range(0,K):
            part0 = G_p_index[:i,:]
            part1 = G_p_index[i:(i+1),:k*M]
            part2 = (tf.matmul(G_w[:,k*L:(k+1)*L]*tf.transpose(lr_w[k*L:(k+1)*L,:]), tf.matmul(H[k*L:(k+1)*L,:], sum1[:,i:(i+1)]))
                     *cof2[k:(k+1),i:(i+1)]*b[k:(k+1),i:(i+1)]*tf.matmul(tf.transpose(tf.conj(w[k*L:(k+1)*L,:])), H[k*L:(k+1)*L,:])
                     +tf.matmul(G_w[:,k*L:(k+1)*L]*tf.transpose(lr_w[k*L:(k+1)*L,:])*cof1[k:(k+1),i:(i+1)]*b[k:(k+1),i:(i+1)], H[k*L:(k+1)*L,:])
                     +tf.matmul(G_p[:,k*M:(k+1)*M]*tf.transpose(lr_p[k*M:(k+1)*M,:]), tf.matmul(tf.transpose(tf.conj(H[k*L:(k+1)*L,:])), w[k*L:(k+1)*L,:])
                                *b[k:(k+1),i:(i+1)])*cof2[k:(k+1),i:(i+1)]*b[k:(k+1),i:(i+1)]*tf.matmul(tf.transpose(tf.conj(w[k*L:(k+1)*L,:])), H[k*L:(k+1)*L,:]))
            part3 = G_p_index[i:(i+1),(k+1)*M:]
            part4 = G_p_index[(i+1):,:]
            G_p_index = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0)

            part0 = G_w_index[:i,:]
            part1 = G_w_index[i:(i+1),:k*L]
            part2 = (tf.matmul(G_w[:,k*L:(k+1)*L]*tf.transpose(lr_w[k*L:(k+1)*L,:]), tf.matmul(H[k*L:(k+1)*L,:], sum1[:,i:(i+1)]))
                     *cof2[k:(k+1),i:(i+1)]*tf.transpose(tf.conj(tf.add(tf.matmul(H[k*L:(k+1)*L,:], sum1[:,i:(i+1)]), n)))
                     +tf.matmul(G_p[:,k*M:(k+1)*M]*tf.transpose(lr_p[k*M:(k+1)*M,:]), tf.matmul(tf.transpose(tf.conj(H[k*L:(k+1)*L,:])), w[k*L:(k+1)*L,:])
                                *b[k:(k+1),i:(i+1)])*cof2[k:(k+1),i:(i+1)]*tf.transpose(tf.conj(tf.add(tf.matmul(H[k*L:(k+1)*L,:], sum1[:,i:(i+1)]), n)))
                     +tf.matmul(G_p[:,k*M:(k+1)*M]*tf.transpose(lr_p[k*M:(k+1)*M,:])*cof1[k:(k+1),i:(i+1)]*b[k:(k+1),i:(i+1)], tf.transpose(tf.conj(H[k*L:(k+1)*L,:]))))
            part3 = G_w_index[i:(i+1),(k+1)*L:]
            part4 = G_w_index[(i+1):,:]
            G_w_index = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0)
    
    G_p_index = tf.reduce_mean(G_p_index, axis=0, keep_dims=True)
    G_w_index = tf.reduce_mean(G_w_index, axis=0, keep_dims=True)
        
    return G_p_index, G_w_index


# In[51]:


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True
sess = tf.InteractiveSession(config=tf_config)
    
delta_ap = tf.complex(tf.zeros(shape=[K*M,layer]), tf.zeros(shape=[K*M,layer]))
delta_aw = tf.complex(tf.zeros(shape=[K*L,layer]), tf.zeros(shape=[K*L,layer]))
for l in range(0,layer):
    p, w, delta_p_avr, delta_w_avr = forward(p,w,lr_p,lr_w)
    
    part0 = delta_ap[:,:l]
    part1 = -delta_p_avr
    part2 = delta_ap[:,(l+1):]
    delta_ap = tf.concat([part0,part1,part2], axis = 1)
    
    part0 = delta_aw[:,:l]
    part1 = -delta_w_avr
    part2 = delta_aw[:,(l+1):]
    delta_aw = tf.concat([part0,part1,part2], axis = 1)
    
    sum1 = tf.complex(tf.zeros(shape=[M,batch_size]), tf.zeros(shape=[M,batch_size]))
    part1 = tf.complex(tf.zeros(shape=[M,1]), tf.zeros(shape=[M,1]))
    for i in range(0,batch_size):
        part0 = sum1[:,0:i]
        part1 = tf.complex(tf.zeros(shape=[M,1]), tf.zeros(shape=[M,1]))
        part2 = sum1[:,(i+1):]
        for k in range(0,K):
            part1 = part1+tf.matmul(p[k*M:(k+1)*M,:], b[k:(k+1),i:(i+1)])
        sum1 = tf.concat([part0,part1,part2], axis=1)
    #print(sum1)
    
    b_hat = tf.complex(tf.zeros(shape=[K,batch_size]), tf.zeros(shape=[K,batch_size]))
    for i in range(0,batch_size):
        for k in range(0,K):
            part0 = b_hat[:k,:]
            part1 = b_hat[k:(k+1),:i]
            part2 = tf.matmul(tf.transpose(tf.conj(w[k*L:(k+1)*L,:])), tf.add(tf.matmul(H[k*L:(k+1)*L,:], sum1[:,i:(i+1)]), n))
            part3 = b_hat[k:(k+1),(i+1):]
            part4 = b_hat[(k+1):,:]
            b_hat = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0)
    
    s = tf.zeros(shape=[K*N,batch_size])
    bit_error = tf.zeros(shape=[1,batch_size])
    
    def pos(s):
        part0 = s[:k,:]
        part1 = s[k:(k+1),:i]
        part2 = tf.ones(shape=[1,1])
        part3 = s[k:(k+1),(i+1):]
        part4 = s[(k+1):,:]
        s = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0), tf.zeros(shape=[K*N,batch_size])
        return s
    def neg(s):
        part0 = s[:k,:]
        part1 = s[k:(k+1),:i]
        part2 = -tf.ones(shape=[1,1])
        part3 = s[k:(k+1),(i+1):]
        part4 = s[(k+1):,:]
        s = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0), tf.zeros(shape=[K*N,batch_size])
        return s
    
    def add_error(bit_error):
        part0 = bit_error[0:1,:i]
        part1 = bit_error[0:1,i:(i+1)]+1
        part2 = bit_error[0:1,(i+1):]
        bit_error = tf.concat([part0,part1,part2], axis=1)
        return bit_error
    def remain_error(bit_error):
        return bit_error
    
    for i in range(0,batch_size):
        for k in range(0,K):
            tf.cond(tf.real(b_hat[k,i]) > 0, lambda: pos(s), lambda: neg(s))
            
            tf.cond(tf.equal(tf.real(b[k,i]), s[k,i]), lambda: remain_error(bit_error), lambda:add_error(bit_error))
                
    
    error_op = error.append(tf.reduce_mean(bit_error).eval())


# In[52]:


def back_grad(delta_p_avr,delta_w_avr):
    G_p = tf.complex(tf.zeros(shape=[layer,K*M]), tf.zeros(shape=[layer,K*M]))
    G_w = tf.complex(tf.zeros(shape=[layer,K*L]), tf.zeros(shape=[layer,K*L]))

    G_p_L = tf.transpose(tf.conj(delta_p_avr))
    G_w_L = tf.transpose(tf.conj(delta_w_avr))
    G_p_index = tf.complex(tf.zeros(shape=[1,K*M]), tf.zeros(shape=[1,K*M]))
    G_w_index = tf.complex(tf.zeros(shape=[1,K*L]), tf.zeros(shape=[1,K*L]))
    #print(G_p_L)

    part0 = G_p[:(layer-1),:]
    part1 = G_p_L
    G_p = tf.concat([part0,part1],axis=0)

    part0 = G_w[:(layer-1),:]
    part1 = G_w_L
    G_w = tf.concat([part0,part1],axis=0)

    G_p_index, G_w_index = back(p,w,G_p_index,G_w_index)
    part0 = G_p[:(layer-2),:]
    part1 = G_p_index
    part2 = G_p[(layer-1):,:]
    G_p = tf.concat([part0,part1,part2],axis=0)

    part0 = G_w[:(layer-2),:]
    part1 = G_w_index
    part2 = G_w[(layer-1):,:]
    G_w = tf.concat([part0,part1,part2],axis=0)

    for l in range(layer-3,-1,-1):
        G_p_index, G_w_index = back(p,w,G_p_index,G_w_index)
        part0 = G_p[:l,:]
        part1 = G_p_index
        part2 = G_p[(l+1):,:]
        G_p = tf.concat([part0,part1,part2],axis=0)

        part0 = G_w[:l,:]
        part1 = G_w_index
        part2 = G_w[(l+1):,:]
        G_w = tf.concat([part0,part1,part2],axis=0)

    global delta_ap
    global delta_aw
    delta_ap1 = tf.transpose(tf.conj(G_p))*delta_ap
    delta_aw1 = tf.transpose(tf.conj(G_w))*delta_aw

    delta_p_layer = tf.transpose(tf.conj(G_p))
    delta_w_layer = tf.transpose(tf.conj(G_w))
    return delta_ap1,delta_aw1,delta_p_layer,delta_w_layer


# In[53]:


global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                              initializer=tf.constant_initializer(0), trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=0.005, global_step=global_step, decay_steps=50000,
                                           decay_rate=0.9, staircase=True)


# In[54]:


delta_ap,delta_aw,delta_p_layer,delta_w_layer = back_grad(delta_p_avr,delta_w_avr)

for l in range(0,layer-1):
    lr_p = lr_p-tf.complex(learning_rate,tf.zeros(shape=[1,1]))*delta_ap[:,l:(l+1)]
    p = p-lr_p*delta_p_layer[:,l:(l+1)]
    lr_w = lr_w-tf.complex(learning_rate,tf.zeros(shape=[1,1]))*delta_aw[:,l:(l+1)]
    w = w-lr_w*delta_w_layer[:,l:(l+1)]

    sum1 = tf.complex(tf.zeros(shape=[M,batch_size]), tf.zeros(shape=[M,batch_size]))
    part1 = tf.complex(tf.zeros(shape=[M,1]), tf.zeros(shape=[M,1]))
    for i in range(0,batch_size):
        part0 = sum1[:,0:i]
        part1 = tf.complex(tf.zeros(shape=[M,1]), tf.zeros(shape=[M,1]))
        part2 = sum1[:,(i+1):]
        for k in range(0,K):
            part1 = part1+tf.matmul(p[k*M:(k+1)*M,:], b[k:(k+1),i:(i+1)])
        sum1 = tf.concat([part0,part1,part2], axis=1)
    #print(sum1)

    b_hat = tf.complex(tf.zeros(shape=[K,batch_size]), tf.zeros(shape=[K,batch_size]))
    for i in range(0,batch_size):
        for k in range(0,K):
            part0 = b_hat[:k,:]
            part1 = b_hat[k:(k+1),:i]
            part2 = tf.matmul(tf.transpose(tf.conj(w[k*L:(k+1)*L,:])), tf.add(tf.matmul(H[k*L:(k+1)*L,:], sum1[:,i:(i+1)]), n))
            part3 = b_hat[k:(k+1),(i+1):]
            part4 = b_hat[(k+1):,:]
            b_hat = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0)

    s = tf.zeros(shape=[K*N,batch_size])
    bit_error = tf.zeros(shape=[1,batch_size])

    def pos(s):
        part0 = s[:k,:]
        part1 = s[k:(k+1),:i]
        part2 = tf.ones(shape=[1,1])
        part3 = s[k:(k+1),(i+1):]
        part4 = s[(k+1):,:]
        s = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0), tf.zeros(shape=[K*N,batch_size])
        return s
    def neg(s):
        part0 = s[:k,:]
        part1 = s[k:(k+1),:i]
        part2 = -tf.ones(shape=[1,1])
        part3 = s[k:(k+1),(i+1):]
        part4 = s[(k+1):,:]
        s = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0), tf.zeros(shape=[K*N,batch_size])
        return s

    def add_error(bit_error):
        part0 = bit_error[0:1,:i]
        part1 = bit_error[0:1,i:(i+1)]+1
        part2 = bit_error[0:1,(i+1):]
        bit_error = tf.concat([part0,part1,part2], axis=1)
        return bit_error
    def remain_error(bit_error):
        return bit_error

    for i in range(0,batch_size):
        for k in range(0,K):
            tf.cond(tf.real(b_hat[k,i]) > 0, lambda: pos(s), lambda: neg(s))

            tf.cond(tf.equal(tf.real(b[k,i]), s[k,i]), lambda: remain_error(bit_error), lambda:add_error(bit_error))


    error1_op = error.append(tf.reduce_mean(bit_error).eval())

p, w, delta_p_avr, delta_w_avr = forward(p,w,lr_p,lr_w)

#part0 = delta_ap[:,:l]
#part1 = -delta_p_avr
#part2 = delta_ap[:,(l+1):]
#delta_ap = tf.concat([part0,part1,part2], axis = 1)
#
#part0 = delta_aw[:,:l]
#part1 = -delta_w_avr
#part2 = delta_aw[:,(l+1):]
#delta_aw = tf.concat([part0,part1,part2], axis = 1)


# In[55]:


input_train_queue = tf.train.slice_input_producer([train_b], shuffle=True)
train_b_batch = tf.train.batch(input_train_queue, batch_size=batch_size, num_threads=1, capacity=32)
train_b_ = tf.complex(tf.transpose(train_b_batch), tf.zeros(shape=[K, batch_size], dtype = tf.float32))
print(train_b_)
    
input_test_queue = tf.train.slice_input_producer([test_b], shuffle=True)
test_b_batch = tf.train.batch(input_test_queue, batch_size=batch_size, num_threads=1, capacity=32)
test_b_ = tf.complex(tf.transpose(test_b_batch), tf.zeros(shape=[K, batch_size], dtype = tf.float32))
print(test_b_)


# In[56]:


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver() 

for i in range(I_max):
    #train_loss_, train_acc_, train_batch_, train_batch_ = 0, 0, 0, 0
    #train_b_list = shuffle_set(train_b)
    for j in range(500):
        #train_b_batch = minibatches(inputs=train_b_list, batch_size=batch_size, now_batch=j, total_batch=500)
        #train_b_ = tf.complex(tf.transpose(train_b_batch), tf.zeros(shape=[K, batch_size], dtype = tf.float32))
        train_batch = sess.run(train_b_)
        #_, step, train_loss, train_acc = sess.run([train_op, global_step, loss, accuracy],
        #                                          feed_dict={input_b:train_batch, is_training:True})
        step = sess.run([global_step], feed_dict={b:train_batch})
        
        #train_loss_ += train_loss
        #train_acc_ += train_acc[1]
        #train_batch_ += 1
    
    #fig_loss1[i] = np.sum(train_loss_)/train_batch_
    #fig_acc1[i] = np.sum(train_acc_)/train_batch_
        
    if i%10 == 0:  # print training 
 #           print(device_lib.list_local_devices())
        print('step:%d , train_error:%.6f' % (i, error[-1]))
    if i%50 == 0:  # save current model
        #save_path = os.path.join('ckpt-mmse-training', 'mser-model.ckpt')
        #saver.save(sess, save_path, global_step=step)
        sio.savemat("training_bit_error.mat", {"array": error})
    #if i%100 == 0: # testing
        #if tf.train.latest_checkpoint('ckpt-mmse-training') is not None:
        #    saver.restore(sess, tf.train.latest_checkpoint('ckpt-mmse-training'))
        #test_loss_, test_acc_, test_batch_, test_batch_ = 0, 0, 0, 0
        #for j in range(150):
            #test_b_list = shuffle_set(test_b)
            #test_b_batch = minibatches(inputs=test_b_list, batch_size=batch_size, now_batch=j, total_batch=150)
            #test_b_ = tf.complex(tf.transpose(test_b_batch), tf.zeros(shape=[K, batch_size], dtype = tf.float32))
            #test_batch = sess.run(test_b_)
            #step = sess.run([global_step], feed_dict={b:test_batch})
            #step, test_loss, test_acc = sess.run([global_step, loss, accuracy],feed_dict={input_b:test_batch, is_training:False})
            #test_loss_ += test_loss
            #test_acc_ += test_acc[1]
            #test_batch_ += 1
        #fig_loss2[tf.cast((i%100),dtype=tf.int32).eval()] = np.sum(test_loss_)/test_batch_
        #fig_acc2[tf.cast((i%100),dtype=tf.int32).eval()] = np.sum(test_acc_)/test_batch_
        
        print('---------------------')
        print('step:%d , test_error:%.6f' % (i, error[-1]))
        print('---------------------')

coord.request_stop()
coord.join(threads)
sess.close()


# In[57]:


sess.close()


# In[ ]:


#sess.run(tf.local_variables_initializer())
#sess.run(tf.global_variables_initializer())
#step= sess.run([global_step], feed_dict={b:train_batch})
sess = tf.InteractiveSession(config=tf_config)
print(p.eval())
#train_error = sess.run(error_op)
#train_error1 = sess.run([error1_op], feed_dict={b:train_batch})


# In[ ]:


pp_ = tf.random_normal(mean=0.0,stddev=tf.sqrt(0.5),shape=[2*K*M, N])
pp = tf.complex(pp_[0:K*M,:], p_[K*M:,:])
kk = tf.complex(tf.constant(1.0), tf.constant(0.0))
def f():
    global pp
    for k in range(0,K):
        part0 = pp[0:k*M,:]
        #part1 = pp[k*M:(k+1)*M,:]
        part2 = pp[(k+1)*M:,:]
        part1 = tf.complex(tf.ones(shape=[M,1]), tf.zeros(shape=[M,1]))
        pp = tf.concat([part0,part1,part2], axis=0)
    return pp

sess = tf.InteractiveSession()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

print(sess.run(pp[0:M,:]))
print(sess.run(f()))


# In[ ]:


sess = tf.InteractiveSession()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
kk1 = (b[:,0:(0+1)]/(2*tf.sqrt(2*3.14*rol))*tf.exp(-tf.complex(tf.pow(tf.real(b_hat[:,0:(0+1)]),2), tf.zeros(shape=[K,1]))/(2*tf.pow(rol,2))))
#kk2 = tf.matmul(tf.transpose(tf.conj(H[0*L:(0+1)*L,:])), w[0*L:(0+1)*L,:])
cof1 = tf.complex(tf.zeros(shape=[K,batch_size]), tf.zeros(shape=[K,batch_size]))
part0 = cof1[:,:0]
part1 = (b[:,0:(0+1)]/(2*tf.sqrt(2*3.14*rol))
         *tf.exp(-tf.complex(tf.pow(tf.real(b_hat[:,0:(0+1)]),2), tf.zeros(shape=[K,1]))/(2*tf.pow(rol,2))))
part2 = cof1[:,(0+1):]
#cof1 = tf.concat([part0,part1,part2], axis=1)
print(kk1)
print(part0)
print(part1)
print(part2)
sess.close()

