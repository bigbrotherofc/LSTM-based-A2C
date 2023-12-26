# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class A2CLSTM(object):
    def __init__(
            self,
            sess,
            n_actions,
            n_features,
            lr_a=0.001,
            lr_c=0.01,
            entropy_beta=0.01 #折扣因子改成了交叉熵
    ):
        self.sess = sess
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.entroy_beta = entropy_beta #折扣因子改成了交叉熵

        self.lstm_cell_size = 64 #LSTM的cell个数多了这个
        #训练优化器
        OPT_A = tf.train.AdamOptimizer(self.lr_a)
        OPT_C = tf.train.AdamOptimizer(self.lr_c)
        #name_scope 命名域，和variable_scope不同，variable_scope是为了共享变量，
        #name_scope是为了更好的管理变量和tf.Variable()和tf.get_variable()都可以创建变量
        """
        tf.placeholder(dtype, shape=None, name=None),状态，动作和
        """
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], "state")
            self.a = tf.placeholder(tf.int32, [None, 1], "action")
            self.td_target = tf.placeholder(tf.float32, [None, 1], "td_target") #r 和 v_变成了这个

        self.acts_prob, self.v, self.a_params, self.c_params = self._build_net() #构建的网络返回值多了

        with tf.name_scope('TD_error'):
            self.td_error = tf.subtract(self.td_target, self.v, name='TD_error') #本质没变

        with tf.name_scope('c_loss'):
            self.c_loss = tf.reduce_mean(tf.square(self.td_error))

        with tf.name_scope('a_loss'): #这个改变了很多
            log_prob = tf.reduce_sum(tf.log(self.acts_prob + 1e-5) * tf.one_hot(self.a, self.n_actions, dtype=tf.float32),
                                     axis=1, keepdims=True)
            exp_v = log_prob * tf.stop_gradient(self.td_error)
            entropy = -tf.reduce_sum(self.acts_prob * tf.log(self.acts_prob + 1e-5), axis=1,
                                     keepdims=True)  # encourage exploration
            self.exp_v = self.entroy_beta * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)

        with tf.name_scope('compute_grads'):
            self.a_grads = tf.gradients(self.a_loss, self.a_params)
            self.c_grads = tf.gradients(self.c_loss, self.c_params)

        with tf.name_scope('c_train'):
            self.c_train_op = OPT_C.apply_gradients(zip(self.c_grads, self.c_params))

        with tf.name_scope('a_train'):
            self.a_train_op = OPT_A.apply_gradients(zip(self.a_grads, self.a_params))

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        b_init = tf.constant_initializer(0.1)

        with tf.variable_scope('Critic'):
            # [time_step, feature] => [time_step, batch, feature]
            s = tf.expand_dims(self.s, axis=1, name='timely_input')

            lstm_cell =  tf.nn.rnn_cell.LSTMCell(self.lstm_cell_size) #状态价值函数加了lstm
            self.lstm_state_init = lstm_cell.zero_state(batch_size=1, dtype=tf.float32)

            outputs, _ = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=s,
                initial_state=self.lstm_state_init,
                time_major=True
            )
            cell_out = tf.reshape(outputs[-1, :, :], [-1, self.lstm_cell_size],
                                  name='flatten_lstm_outputs')  # joined state representation

            l_c1 = tf.layers.dense(
                inputs=cell_out,
                units=32,
                activation=tf.nn.tanh,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                name='l_c1'
            )

            v = tf.layers.dense(
                inputs=l_c1,
                units=1,
                kernel_initializer=w_init,
                bias_initializer=b_init,
                name='V'
            )  # state value
        # Actor没变
        with tf.variable_scope('Actor'):
            l_a1 = tf.layers.dense(
                inputs=cell_out, #对 ，就是输入变成了 就是在状态层前面加了一个LSTM的输出，你在LSTM前面还可以加。
                units=32,  # number of hidden units
                activation=tf.nn.tanh,
                kernel_initializer=w_init,  # weights
                bias_initializer=b_init,  # biases
                name='l_a1'
            )
            #动作预测，这动作空间大的一批啊
            acts_prob = tf.layers.dense(
                inputs=l_a1,
                units=self.n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=w_init,  # weights
                name='acts_prob'
            )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        return acts_prob, v, a_params, c_params

    def choose_action(self, s):
        probs = self.sess.run(self.acts_prob, feed_dict={self.s: s})  # get probabilities for all actions
        a = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        return a

    def learn(self, feed_dict):
        self.sess.run([self.a_train_op, self.c_train_op], feed_dict=feed_dict)

    def target_v(self, s):
        v = self.sess.run(self.v, {self.s: s})
        return v
