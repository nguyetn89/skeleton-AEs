''' Estimating skeleton-based gait abnormality index by sparse deep auto-encoder
    BSD 2-Clause "Simplified" License
    Author: Trong-Nguyen Nguyen'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

from utils import *

'''=================== DATASET ======================'''
dataset = 'dataset/DIRO_skeletons.npz'
loaded = np.load(dataset)
skel_data = loaded['data']
separation = loaded['split']

n_full_joint = skel_data.shape[-1]//3
assert n_full_joint == 25

n_gait, n_frame = skel_data.shape[1:3]

training_subjects = np.where(separation == 'train')[0]
test_subjects = np.where(separation == 'test')[0]
n_test_subject = len(test_subjects)

joints = [3, 20, 4, 8, 5, 9, 7, 11, 0, 12, 16, 13, 17, 14, 18, 15, 19] # selected joints
n_joint = len(joints)

'''=================== NETWORK ======================'''
tf.reset_default_graph()
tf.set_random_seed(2018)
np.random.seed(2018)

n_epoch = 256
batchsize = 100

def dense(x, n_in, n_out, scope=None, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [n_in, n_out], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [n_out], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias

def kl_divergence(rho_hat, rho = 0.05, fuzz = 1e-8):
    return rho * tf.log(rho) - rho * tf.log(rho_hat + fuzz) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat + fuzz)

def autoencoder(input_tensor):
    with tf.variable_scope("autodecoder"):
        t0, w0, b0 = dense(input_tensor, n_joint, 128, scope = "ae_h0", with_w = True)
        h0 = tf.nn.sigmoid(t0) # 17 -> 128 (with sparsity)
        t1, w1, b1 = dense(h0, 128, 32, scope = "ae_h1", with_w = True)
        h1 = tf.nn.tanh(t1) # 128 -> 32
        t3, w3, b3 = dense(h1, 32, 8, scope = "ae_h3", with_w = True)
        h3 = tf.nn.tanh(t3) # 32 -> 8
        t4, w4, b4 = dense(h3, 8, 32, scope = "ae_h4", with_w = True)
        h4 = tf.nn.tanh(t4) # 8 -> 32
        t6, w6, b6 = dense(h4, 32, 128, scope = "ae_h6", with_w = True)
        h6 = tf.nn.tanh(t6) # 32 -> 128
        t7, w7, b7 = dense(h6, 128, 17, scope = 'ae_h7', with_w = True)
        h7 = tf.nn.sigmoid(t7) # 128 -> 17
    return h7, h3, h0, w0, w1, w3, w4, w6, w7 # output, latent code, first hidden layer, weights...

skels = tf.placeholder(tf.float32, [None, n_joint])
r_skels, latent_codes, first_deep_layer, w0, w1, w3, w4, w6, w7 = autoencoder(skels)
#
rho_hat = tf.reduce_mean(first_deep_layer,axis=0)
kl = kl_divergence(rho_hat)
#
decoding_loss = 0.5 * tf.reduce_mean(tf.reduce_sum((skels-r_skels)**2,axis=1))
L2_regularization = 0.5 * (tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(w6) + tf.nn.l2_loss(w7))
KL_loss = tf.reduce_sum(kl)
alpha = 0.2
beta = 1
cost = tf.reduce_mean(decoding_loss) + alpha * L2_regularization + beta * KL_loss
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

def train(axis):
    if axis in ('x', 'X'):
        train_data = train_data_X
        test_data_normal = test_data_normal_X
        test_data_abnormal = test_data_abnormal_X
    elif axis in ('y', 'Y'):
        train_data = train_data_Y
        test_data_normal = test_data_normal_Y
        test_data_abnormal = test_data_abnormal_Y
    elif axis in ('z', 'Z'):
        train_data = train_data_Z
        test_data_normal = test_data_normal_Z
        test_data_abnormal = test_data_abnormal_Z
    else:
        print('unknown axis')
        return None, None, None
    losses = np.array([])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epoch):
            for idx in range(int(n_training_samples / batchsize)):
                batch = train_data[idx*batchsize:(idx+1)*batchsize]
                _, dec_loss, kl_loss, l2_w, total_loss = sess.run((optimizer, decoding_loss, KL_loss, L2_regularization, cost), feed_dict={skels: batch})
                losses = np.append(losses, total_loss)
        #
        _r_skels, _, _, weight0 = sess.run((r_skels, latent_codes, first_deep_layer, w0), feed_dict={skels: train_data})
        diff_data_train = tf.reduce_sum((train_data-_r_skels)**2,axis=1)
        _r_skels, _, _ = sess.run((r_skels, latent_codes, first_deep_layer), feed_dict={skels: test_data_normal})
        diff_data_normal = tf.reduce_sum((test_data_normal-_r_skels)**2,axis=1)
        _r_skels, _, _ = sess.run((r_skels, latent_codes, first_deep_layer), feed_dict={skels: test_data_abnormal})
        diff_data_abnormal = tf.reduce_sum((test_data_abnormal-_r_skels)**2,axis=1)
        return diff_data_train.eval(), diff_data_normal.eval(), diff_data_abnormal.eval(), losses, weight0

'''=================== ASSESSMENT ======================'''
train_data_X, train_data_Y, train_data_Z, \
    test_data_normal_X, test_data_normal_Y, test_data_normal_Z, \
    test_data_abnormal_X, test_data_abnormal_Y, test_data_abnormal_Z = \
        loadskel(skel_data, training_subjects, joints, n_joint, batchsize)

n_training_samples = train_data_X.shape[0]
print('')

print('Training X...')
err_train_X, err_data_normal_X, err_data_abnormal_X, losses_train_X, w0_X = train('X')
assessment_full(err_data_abnormal_X, err_data_normal_X, 1)

print('Training Y...')
err_train_Y, err_data_normal_Y, err_data_abnormal_Y, losses_train_Y, w0_Y = train('Y')
assessment_full(err_data_abnormal_Y, err_data_normal_Y, 1)

print('Training Z...')
err_train_Z, err_data_normal_Z, err_data_abnormal_Z, losses_train_Z, w0_Z = train('Z')
assessment_full(err_data_abnormal_Z, err_data_normal_Z, 1)

'''measure summation'''
seg_lens = [1, 20, n_frame]

print('\nsimple sum:')
sum_abnormal = err_data_abnormal_X + err_data_abnormal_Y + err_data_abnormal_Z
sum_normal = err_data_normal_X + err_data_normal_Y + err_data_normal_Z

assessment_full(sum_abnormal, sum_normal, seg_lens)

print('\nweighted sum:')
err_train_X = np.mean(err_train_X)
err_train_Y = np.mean(err_train_Y)
err_train_Z = np.mean(err_train_Z)
w_X = (err_train_X + err_train_Y + err_train_Z) / err_train_X
w_Y = (err_train_X + err_train_Y + err_train_Z) / err_train_Y
w_Z = (err_train_X + err_train_Y + err_train_Z) / err_train_Z
w_sum_abnormal = w_X * err_data_abnormal_X + w_Y * err_data_abnormal_Y + w_Z * err_data_abnormal_Z
w_sum_normal = w_X * err_data_normal_X + w_Y * err_data_normal_Y + w_Z * err_data_normal_Z

assessment_full(w_sum_abnormal, w_sum_normal, seg_lens)
