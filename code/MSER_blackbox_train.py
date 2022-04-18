#!/home/jiangzhihui/.conda/envs/tensorflow/bin/python3.6
# coding: utf-8

# In[8]:

print("111111111111111111111111")

import tensorflow
import tensorflow as tf
import numpy as np
import os
import math
import scipy.io as sio
#from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt

#tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph() 

# In[9]:


def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)


# In[10]:


# 实现Batch Normalization
def bn_layer(x, training, name):
    # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
    shape = x.shape
    assert len(shape) in [2,4]

    param_shape = shape[-1]
    with tf.variable_scope(name):
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
        gamma = tf.get_variable('gamma',param_shape,initializer=tf.constant_initializer(1))
        beta  = tf.get_variable('beta', param_shape,initializer=tf.constant_initializer(0))
        
        pop_mean = tf.get_variable('pop_mean', param_shape,initializer=tf.constant_initializer(0), trainable=False)
        pop_variance = tf.get_variable('pop_var', param_shape,initializer=tf.constant_initializer(1), trainable=False)

        # 计算当前整个batch的均值与方差
        if len(shape)==4:
            axes = list(range(len(shape)-1))
            batch_mean, batch_variance = tf.nn.moments(x,axes,name='moments')
        else:
            batch_mean, batch_variance = tf.nn.moments(x,[1],name='moments')
        
        def batch_norm_training():
            decay = 0.9
            train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
            train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))
            
            with tf.control_dependencies([train_mean, train_variance]):
                return tf.nn.batch_normalization(x, batch_mean, batch_variance, beta, gamma, 1e-5)
        
        def batch_norm_inference():
            return tf.nn.batch_normalization(x, pop_mean, pop_variance, beta, gamma, 1e-5)
            
        batch_normalized_output = tf.cond(tf.equal(training,True), batch_norm_training, batch_norm_inference)
        
        return batch_normalized_output
    
        ## 采用滑动平均更新均值与方差
        #ema = tf.train.ExponentialMovingAverage(decay=0.9)
        #
        #def mean_var_with_update():
        #    ema_apply_op = ema.apply([batch_mean,batch_variance])
        #    with tf.control_dependencies([ema_apply_op]):
        #        return tf.identity(batch_mean), tf.identity(batch_variance)
        #
        ## 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
        #mean, variance = tf.cond(tf.equal(training,True),mean_var_with_update,
        #        lambda:(ema.average(batch_mean),ema.average(batch_variance)))
        #
        ## 最后执行batch normalization
        #return tf.nn.batch_normalization(x,mean,variance,beta,gamma,1e-5)


# In[11]:


L, M, N, K, snr = 5, 16, 1, 10, 5
batch_size ,interations, n_batches, test_batches = 50, 3000, 200, 60
#p_ = tf.random_normal(mean=0.0,stddev=tf.sqrt(0.5),shape=[2*K*M, N])
#w_ = tf.random_normal(mean=0.0,stddev=tf.sqrt(0.5),shape=[2*K*L, N])
n = tf.random_normal(mean=0.0,stddev=tf.sqrt(0.5*math.pow(10, -snr/10)),shape=[2*L, 1])
input_b = tf.placeholder(dtype=tf.complex64, shape=[K,batch_size])
H = tf.random_normal(mean=0.0,stddev=tf.sqrt(0.5),shape=[K*L, 2*M])
is_training = tf.placeholder(dtype=tf.bool)

train_b = np.sign(np.random.uniform(-1, 1, (10000, K)))
test_b = np.sign(np.random.uniform(-1, 1, (3000, K)))


# In[12]:


if __name__ == '__main__':
    H1 = tf.pad(H,[[int((64-K*L)/2),int((64-K*L)/2)],[int((64-2*M)/2),int((64-2*M)/2)]],"CONSTANT")
    H_reshape = tf.reshape(H1, [-1, 64, 64, 1])
    with tf.variable_scope("p-layer1-conv1"):
        p_conv1 = tf.layers.conv2d(H_reshape, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2020),
                                   name='p_conv')
        #p_bn1 = tf.layers.batch_normalization(p_conv1, training=is_training, name='p_bn1')
        p_bn1 = bn_layer(p_conv1, training=is_training, name='p_bn')
        p_act1 = LeakyRelu(p_bn1, name = 'p_act')
        p_pool1 = tf.layers.max_pooling2d(p_act1, pool_size=[2, 2], strides=[2, 2], padding='same', name='p_pool')
    
    with tf.variable_scope("p-layer2-conv2"):
        p_conv2 = tf.layers.conv2d(p_pool1, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2020),
                                   name='p_conv')
        #p_bn2 = tf.layers.batch_normalization(p_conv2, training=is_training, name='p_bn2')
        p_bn2 = bn_layer(p_conv2, training=is_training, name='p_bn')
        p_act2 = LeakyRelu(p_bn2, name = 'p_act')
        p_pool2 = tf.layers.max_pooling2d(p_act2, pool_size=[2, 2], strides=[2, 2], padding='same', name='p_pool')
    
    with tf.variable_scope("p-layer3-conv3"):
        p_conv3 = tf.layers.conv2d(p_pool2, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2020),
                                   name='p_conv')
        p_bn3 = bn_layer(p_conv3, training=is_training, name='p_bn')
        p_act3 = LeakyRelu(p_bn3, name = 'p_act')
        p_pool3 = tf.layers.max_pooling2d(p_act3, pool_size=[2, 2], strides=[2, 2], padding='same', name='p_pool')
    
    with tf.variable_scope("p-layer4-conv4"):
        p_conv4 = tf.layers.conv2d(p_pool3, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2020),
                                   name='p_conv')
        p_bn4 = bn_layer(p_conv4, training=is_training, name='p_bn')
        p_act4 = LeakyRelu(p_bn4, name = 'p_act')
        p_pool4 = tf.layers.max_pooling2d(p_act4, pool_size=[2, 2], strides=[2, 2], padding='same', name='p_pool')
    
    
    p_flatten_layer = tf.contrib.layers.flatten(p_pool4, 'p_flatten_layer')
    #p_weights1 = tf.get_variable(shape=[p_flatten_layer.shape[-1], 1024], dtype=tf.float32,
    #                             initializer=tf.truncated_normal_initializer(stddev=0.1), name='p_fc_weights1')
    #p_biases1 = tf.get_variable(shape=[1024], dtype=tf.float32,
    #                            initializer=tf.constant_initializer(0.0), name='p_fc_biases1')
    #p_dense1 = tf.nn.bias_add(tf.matmul(p_flatten_layer, p_weights1), p_biases1, name='p_dense1')
    with tf.variable_scope("p-layer5-fc1"):
        p_dense1 = tf.layers.dense(inputs = p_flatten_layer, units = 1024, 
                                   activation = None,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer(seed=2020),
                                   name = 'p_dense')
        #p_bn3 = tf.layers.batch_normalization(p_dense1, training=is_training, name='p_bn3')
        p_bn5 = bn_layer(p_dense1, training=is_training, name='p_bn')
        p_act5 = LeakyRelu(p_bn5, name = 'p_act')
    print(p_dense1)
    
    with tf.variable_scope("p-layer6-fc2"):
        p_dense2 = tf.layers.dense(inputs = p_act5, units = 512, 
                                   activation = None,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer(seed=2020),
                                   name = 'p_dense')
        p_bn6 = bn_layer(p_dense2, training=is_training, name='p_bn')
        p_act6 = LeakyRelu(p_bn6, name = 'p_act')
    print(p_dense2)
    
    #p_weights2 = tf.get_variable(shape=[p_act4.shape[-1], 2*M*N], dtype=tf.float32,
    #                             initializer=tf.truncated_normal_initializer(stddev=0.1), name='p_fc_weights2')
    #p_biases2 = tf.get_variable(shape=[2*M*N], dtype=tf.float32,
    #                            initializer=tf.constant_initializer(0.0), name='p_fc_biases2')
    #p_dense2 = tf.nn.bias_add(tf.matmul(p_act4, p_weights2), p_biases2, name='p_dense2')
    with tf.variable_scope("p-layer7-fc3"):
        p_dense3 = tf.layers.dense(inputs = p_act6, units = 2*K*M*N, 
                                   activation = None,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer(seed=2020),
                                   name = 'p_dense')
        p_bn7 = bn_layer(p_dense3, training=is_training, name='p_bn')
        #p_logit_output = tf.layers.batch_normalization(p_dense2, training=is_training, name='p_logit_output')
        p_logit_output = LeakyRelu(p_bn7, name='p_logit_output')
        #p_logit_output = tf.nn.softmax(p_dense2, name='p_logit_output')
    
    p_ = tf.reshape(p_logit_output, [2*K*M,N])
    real_p = p_[0:K*M,:]
    imag_p = p_[K*M:,:]
    p = tf.complex(real_p, imag_p)
    print(p)
    
    
    with tf.variable_scope("w-layer1-conv1"):
        w_conv1 = tf.layers.conv2d(H_reshape, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2020),
                                   name='w_conv')
        #w_bn1 = tf.layers.batch_normalization(w_conv1, training=is_training, name='w_bn1')
        w_bn1 = bn_layer(w_conv1, training=is_training, name='w_bn')
        w_act1 = LeakyRelu(w_bn1, name = 'w_act')
        w_pool1 = tf.layers.max_pooling2d(w_act1, pool_size=[2, 2], strides=[2, 2], padding='same', name='w_pool')
    
    with tf.variable_scope("w-layer2-conv2"):
        w_conv2 = tf.layers.conv2d(w_pool1, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2020),
                                   name='w_conv')
        #w_bn2 = tf.layers.batch_normalization(w_conv2, training=is_training, name='w_bn2')
        w_bn2 = bn_layer(w_conv2, training=is_training, name='w_bn')
        w_act2 = LeakyRelu(w_bn2, name = 'w_act')
        w_pool2 = tf.layers.max_pooling2d(w_act2, pool_size=[2, 2], strides=[2, 2], padding='same', name='w_pool')
    
    with tf.variable_scope("w-layer3-conv3"):
        w_conv3 = tf.layers.conv2d(w_pool2, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2020),
                                   name='w_conv')
        w_bn3 = bn_layer(w_conv3, training=is_training, name='w_bn')
        w_act3 = LeakyRelu(w_bn3, name = 'w_act')
        w_pool3 = tf.layers.max_pooling2d(w_act3, pool_size=[2, 2], strides=[2, 2], padding='same', name='w_pool')
    
    with tf.variable_scope("w-layer4-conv4"):
        w_conv4 = tf.layers.conv2d(w_pool3, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2020),
                                   name='w_conv')
        w_bn4 = bn_layer(w_conv4, training=is_training, name='w_bn')
        w_act4 = LeakyRelu(w_bn4, name = 'w_act')
        w_pool4 = tf.layers.max_pooling2d(w_act4, pool_size=[2, 2], strides=[2, 2], padding='same', name='w_pool')
    
    
    w_flatten_layer = tf.contrib.layers.flatten(w_pool4, 'w_flatten_layer')
    #w_weights1 = tf.get_variable(shape=[w_flatten_layer.shape[-1], 1024], dtype=tf.float32,
    #                             initializer=tf.truncated_normal_initializer(stddev=0.1), name='w_fc_weights1')
    #w_biases1 = tf.get_variable(shape=[1024], dtype=tf.float32,
    #                            initializer=tf.constant_initializer(0.0), name='w_fc_biases1')
    #w_dense1 = tf.nn.bias_add(tf.matmul(w_flatten_layer, w_weights1), w_biases1, name='w_dense1')
    with tf.variable_scope("w-layer5-fc1"):
        w_dense1 = tf.layers.dense(inputs = w_flatten_layer, units = 1024, 
                                   activation = None,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer(seed=2020),
                                   name = 'w_dense')
        #w_bn3 = tf.layers.batch_normalization(w_dense1, training=is_training, name='w_bn3')
        w_bn5 = bn_layer(w_dense1, training=is_training, name='w_bn')
        w_act5 = LeakyRelu(w_bn5, name = 'w_act')
    print(w_dense1)
    
    with tf.variable_scope("w-layer6-fc2"):
        w_dense2 = tf.layers.dense(inputs = w_act5, units = 512, 
                                   activation = None,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer(seed=2020),
                                   name = 'w_dense')
        w_bn6 = bn_layer(w_dense2, training=is_training, name='w_bn')
        w_act6 = LeakyRelu(w_bn6, name = 'w_act')
    print(w_dense2)
    
    #w_weights2 = tf.get_variable(shape=[w_act4.shape[-1], 2*L*N], dtype=tf.float32,
    #                             initializer=tf.truncated_normal_initializer(stddev=0.1), name='w_fc_weights2')
    #w_biases2 = tf.get_variable(shape=[2*L*N], dtype=tf.float32,
    #                            initializer=tf.constant_initializer(0.0), name='w_fc_biases2')
    #w_dense2 = tf.nn.bias_add(tf.matmul(w_act4, w_weights2), w_biases2, name='w_dense2')
    with tf.variable_scope("w-layer7-fc3"):
        w_dense3 = tf.layers.dense(inputs = w_act6, units = 2*K*L*N, 
                                   activation = None,
                                   kernel_initializer = tf.contrib.layers.xavier_initializer(seed=2020),
                                   name = 'w_dense')
        w_bn7 = bn_layer(w_dense3, training=is_training, name='w_bn')
        #w_logit_output = tf.layers.batch_normalization(w_dense2, training=is_training, name='w_logit_output')
        w_logit_output =  LeakyRelu(w_bn7, name='w_logit_output')
        #w_logit_output = tf.nn.softmax(w_dense2, name='w_logit_output')
    
    w_ = tf.reshape(w_logit_output, [2*K*L,N])
    real_w = w_[0:K*L,:]
    imag_w = w_[K*L:,:]
    w = tf.complex(real_w, imag_w)
    print(w)


# In[13]:


real_H = H[:,0:M]
imag_H = H[:,M:]
H_ = tf.complex(real_H, imag_H)
real_n = n[0:L,:]
imag_n = n[L:,:]
n_ = tf.complex(real_n, imag_n)

sum2 = tf.complex(tf.zeros(shape=[M,M]), tf.zeros(shape=[M,M]))
rol = tf.complex(tf.zeros(shape=[K,1]), tf.zeros(shape=[K,1]))
for k in range(0,K):
    sum2 = sum2+tf.matmul(p[k*M:(k+1)*M,:], tf.transpose(tf.conj(p[k*M:(k+1)*M,:])))
for k in range(0,K):
    part0 = rol[0:k,:]
    part1 = tf.sqrt(tf.matmul(tf.matmul(tf.matmul(tf.matmul(tf.transpose(tf.conj(w[k*L:(k+1)*L,:])), H_[k*L:(k+1)*L,:]), sum2),
                                        tf.transpose(tf.conj(H_[k*L:(k+1)*L,:]))), w[k*L:(k+1)*L,:]))
    part2 = rol[k+1:,:]
    rol = tf.concat([part0,part1,part2], axis=0)

b_hat = tf.complex(tf.zeros(shape=[K,batch_size]), tf.zeros(shape=[K,batch_size]))
sum1 = tf.complex(tf.zeros(shape=[M,batch_size]), tf.zeros(shape=[M,batch_size]))
for i in range(0,batch_size):
    part0 = sum1[:,0:i]
    part1 = tf.complex(tf.zeros(shape=[M,1]), tf.zeros(shape=[M,1]))
    part2 = sum1[:,(i+1):]
    for k in range(0,K):
        part1 = part1+tf.matmul(p[k*M:(k+1)*M,:], input_b[k:(k+1),i:(i+1)])
    sum1 = tf.concat([part0,part1,part2], axis=1)
print(sum1)

for i in range(0,batch_size):
    for k in range(0,K):
        part0 = b_hat[:k,:]
        part1 = b_hat[k:(k+1),:i]
        part2 = tf.matmul(tf.transpose(tf.conj(w[k*L:(k+1)*L,:])), tf.add(tf.matmul(H_[k*L:(k+1)*L,:], sum1[:,i:(i+1)]), n_))
        part3 = b_hat[k:(k+1),(i+1):]
        part4 = b_hat[(k+1):,:]
        b_hat = tf.concat([part0,tf.concat([part1,part2,part3], axis=1),part4], axis=0)
print(b_hat)


# In[14]:


sess = tf.InteractiveSession()
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True
sess = tf.InteractiveSession(config=tf_config)
#sess = tf.InteractiveSession()

def f(x):
    return tf.exp(-tf.pow(x,2))

def asr(a, b, n):
    h = (b-a) / (2*n)
    F0 = f(a) + f(b)
    F1 = 0
    F2 = 0
    for j in range(1, tf.cast((2*n),dtype=tf.int32).eval()):
        x = a + (j*h)
        if j%2 == 0:
            F2 = F2 + f(x)
        else:
            F1 = F1 + f(x)
    SN = (h * (F0 + 2*F2 + 4*F1))/3.0
    return SN


# In[15]:


loss1 = -tf.real(b_hat)*tf.real(input_b)/(tf.sqrt(2.0)*tf.real(rol))
integ = asr(-tf.cast(tf.pow(10, 3), dtype = tf.float32), loss1, tf.cast(tf.constant(500), dtype = tf.float32))
loss = 1/tf.sqrt(3.14)*tf.reduce_mean(integ)
print(loss1)

accuracy = tf.metrics.accuracy(
    labels = input_b,
    predictions = tf.sign(tf.real(b_hat))
)

global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                              initializer=tf.constant_initializer(0), trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=0.005, global_step=global_step, decay_steps=50000,
                                           decay_rate=0.9, staircase=True)
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='optimizer')
#with tf.control_dependencies(update_ops):
#grads = opt.compute_gradients(loss)
train_op = opt.minimize(loss, global_step=global_step)


# In[16]:


#input_train_queue = tf.train.slice_input_producer([train_b], shuffle=True)
#train_b_batch = tf.train.batch(input_train_queue, batch_size=batch_size, num_threads=1, capacity=32)
#train_b_ = tf.complex(tf.transpose(train_b_batch), tf.zeros(shape=[K, batch_size], dtype = tf.float32))
#print(train_b_)
#
#input_test_queue = tf.train.slice_input_producer([test_b], shuffle=True)
#test_b_batch = tf.train.batch(input_test_queue, batch_size=batch_size, num_threads=1, capacity=32)
#test_b_ = tf.complex(tf.transpose(test_b_batch), tf.zeros(shape=[K, batch_size], dtype = tf.float32))
#print(test_b_)
#def shuffle_set(inputs):
#indices = np.arange(tf.cast(inputs.get_shape()[0], dtype=tf.int32).eval())
#np.random.shuffle(inputs.eval())
#inputs = inputs[indices]
#return inputs

'''
def minibatches(inputs, batch_size, now_batch, total_batch):
if now_batch < total_batch-1:
    j_batch = inputs[now_batch*batch_size:(now_batch+1)*batch_size,:]
else:
    j_batch = inputs[now_batch*batch_size:,:]
return j_batch
'''

# In[ ]:


#coord = tf.train.Coordinator()
#threads = tf.train.start_queue_runners(sess, coord)
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver() 

fig_loss1 = []
fig_loss2 = []
fig_acc1 = []
fig_acc2 = []

#if tf.train.latest_checkpoint('ckpt-mmse-training') is not None:
#model_dir = "ckpt-mmse-training"
#last_path = os.path.join(model_dir, "mser-model.ckpt-578500")
#saver.restore(sess, last_path)
for i in range(interations):
    train_loss_, train_acc_, train_batch_ = 0, 0, 0
    train_b_list = np.random.permutation(train_b)
    for j in range(n_batches):
        #train_b_batch = minibatches(inputs=train_b_list, batch_size=batch_size, now_batch=j, total_batch=500)
        train_b_batch = np.matrix(train_b_list)[j*batch_size:(j+1)*batch_size,:]
        train_batch = train_b_batch.T+1j*np.zeros((K, batch_size))
        #train_batch = sess.run(train_b_)
        _, step, train_loss, train_acc = sess.run([train_op, global_step, loss, accuracy],
                                                  feed_dict={input_b:train_batch, is_training:True})
        train_loss_ += train_loss
        train_acc_ += train_acc[1]
        train_batch_ += 1
    
    fig_loss1.append(np.sum(train_loss_)/train_batch_)
    fig_acc1.append(np.sum(train_acc_)/train_batch_)
        
    if i%10 == 0:  # print training 
 #           print(device_lib.list_local_devices())
        print('step:%d , train_loss:%.6f , train_accuracy:%.6f' % (i, fig_loss1[-1], fig_acc1[-1]))
    if i%50 == 0:  # save current model
        save_path = os.path.join('ckpt-mmse-training_M16', 'mser-model.ckpt')
        saver.save(sess, save_path, global_step=step)
    if i%100 == 0: # testing
        if tf.train.latest_checkpoint('ckpt-mmse-training_M16') is not None:
            saver.restore(sess, tf.train.latest_checkpoint('ckpt-mmse-training_M16'))
        test_loss_, test_acc_, test_batch_ = 0, 0, 0
        for j in range(test_batches):
            test_b_list = np.random.permutation(test_b)
            #test_b_batch = minibatches(inputs=test_b_list, batch_size=batch_size, now_batch=j, total_batch=150)
            test_b_batch = np.matrix(test_b_list)[j*batch_size:(j+1)*batch_size,:]
            test_batch = test_b_batch.T+1j*np.zeros((K, batch_size))
            #test_batch = sess.run(test_b_)
            step, test_loss, test_acc = sess.run([global_step, loss, accuracy],feed_dict={input_b:test_batch, is_training:False})
            test_loss_ += test_loss
            test_acc_ += test_acc[1]
            test_batch_ += 1
        fig_loss2.append(np.sum(test_loss_)/test_batch_)
        fig_acc2.append(np.sum(test_acc_)/test_batch_)
        
        print('---------------------')
        print('step:%d , test_loss:%.6f , test_accuracy:%.6f' % (i, test_loss, test_acc[1]))
        print('---------------------')

#coord.request_stop()
#coord.join(threads)
sess.close()

sio.savemat("train_loss_M16_5db.mat", {"train_loss": fig_loss1})
sio.savemat("train_acc_M16_5db.mat", {"train_acc": fig_acc1})
sio.savemat("test_loss_M16_5db.mat", {"train_loss": fig_loss2})
sio.savemat("test_acc_M16_5db.mat", {"train_acc": fig_acc2})
# In[ ]:


#ll1 = tf.complex(tf.real(b_hat)*input_b,0.0)
#ll2 = tf.complex(tf.sqrt(2.0),0.0)*rol
#loss1 = -tf.real(tf.complex(tf.real(b_hat)*input_b,0.0)/(tf.complex(tf.sqrt(2.0),0.0)*rol))


# In[6]:


#print(sess.run(learning_rate))
#print("------------")
#print(sess.run(w_bn4, {input_b:train_batch ,is_training:True}))
#print("------------")
#print(sess.run(w_dense2, {input_b:train_batch ,is_training:True}))
#print("------------")
#print(sess.run(w_logit_output))
#print("------------")
#print(sess.run(w_act3, {input_b:train_batch ,is_training:True}))
#print("------------")
#print(sess.run(p, {input_b:train_batch ,is_training:True}))
#print("------------")
#print(sess.run(w, {input_b:train_batch ,is_training:True}))
#print("------------")
#print(fig_loss1)
#print("------------")
#print(b1.get_shape())
#print(b_hat.get_shape())


# In[4]:


#from tensorflow.python import pywrap_tensorflow
#model_dir = "ckpt-mmse-training"
#checkpoint_path = os.path.join(model_dir, "mser-model.ckpt-225500")
#reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
#var_to_shape_map = reader.get_variable_to_shape_map()
#for key in var_to_shape_map:
#    if 'fig_loss' in key:
#        print("tensor_name: ", key, end=' ')
#        print(reader.get_tensor(key), end="\n")


# In[7]:

'''
fig2=plt.figure()
ax2=fig2.add_subplot(111)
ax2.set_title('CNN_Accuracy',fontsize=16)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.plot(np.arange(1,interations+1), fig_acc1,label="Train Acc")
ax2.plot(np.arange(1,interations+1), fig_acc2,label="Test Acc")
ax2.legend(loc=4)
x_values=range(1,interations+1)
ax2.set_xticks(x_values[::1])
#ax2.grid(linestyle='--', alpha=0.8)

fig1=plt.figure()
ax1=fig1.add_subplot(111)
ax1.plot(np.arange(1,interations+1), fig_loss1,label="Train Loss")
ax1.plot(np.arange(1,interations+1), fig_loss2,label="Test Loss")
ax1.legend(loc=1)#第一象限
ax1.set_title('CNN_Loss',fontsize=16)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
x_values=range(1,interations+1)
ax1.set_xticks(x_values[::1])
#ax1.grid(linestyle='--', alpha=0.8)

fig1.savefig('C:/Users/shishuhan/MSER/cnnloss.png')
fig2.savefig('C:/Users/shishuhan/MSER/cnnacc.png')
plt.show()
'''

# In[ ]:




