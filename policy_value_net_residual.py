# -*- coding: utf-8 -*-
"""
An implementation of the policy value network in Tensorflow
Residual Network, batch normalization
Not completed

Tensorflow v1.6

@author: Wen Liang
"""

import numpy as np
import tensorflow as tf


class PolicyValueNet():
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        n_filter_out = 64

        # Define the tensorflow neural network
        # 1. Input:
        self.input_states = tf.placeholder(
                tf.float32, shape=[None, 4, board_height, board_width])
        self.input_states_reshaped = tf.reshape(
                self.input_states, [-1, board_height, board_width, 4])
        # 2. Common Networks Layers
        self.conv1 = tf.layers.conv2d(inputs=self.input_states_reshaped,
                                      filters=64, kernel_size=[5, 5],
                                      padding="same", activation=None)
        self.bn1 = tf.contrib.layers.batch_norm(self.conv1,
                                                center=True, scale=True, is_training=True)
        self.conv1 = tf.nn.relu(self.bn1)

        # 3.1 residual blocks 1
        self.block1 = self.residual_block(self.conv1, 64)
        self.block2 = self.residual_block(self.block1, 64)

        # # block2
        # self.conv31 = tf.layers.conv2d(inputs=self.block1, filters=64,
        #                                kernel_size=[3, 3], padding="same",
        #                                activation=None)
        # self.bn31 = tf.contrib.layers.batch_norm(self.conv31,
        #                                          center=True, scale=True, is_training=True)
        # self.conv31 = tf.nn.relu(self.bn31)
        # self.conv32 = tf.layers.conv2d(inputs=self.conv31, filters=64,
        #                                kernel_size=[3, 3], padding="same",
        #                                activation=None)
        # self.bn22 = tf.contrib.layers.batch_norm(self.conv22,
        #                                          center=True, scale=True, is_training=True)
        # self.conv22 = tf.nn.relu(self.bn22)
        # self.block1 = self.conv1 + self.conv22



        # 2. Common Networks Layers my parameters
        # self.conv1 = tf.layers.conv2d(inputs=self.input_states_reshaped,
        #                               filters=32, kernel_size=[5, 5],
        #                               padding="same", activation=tf.nn.relu)
        # self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
        #                               kernel_size=[5, 5], padding="same",
        #                               activation=tf.nn.relu)
        # self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
        #                               kernel_size=[5, 5], padding="same",
        #                               activation=tf.nn.relu)
        # 3-1 Action Networks
        self.action_conv = tf.layers.conv2d(inputs=self.block2, filters=n_filter_out,
                                            kernel_size=[1, 1], padding="same",
                                            activation=tf.nn.relu)
        # Flatten the tensor
        self.action_conv_flat = tf.reshape(
                self.action_conv, [-1, n_filter_out * board_height * board_width])
        # 3-2 Full connected layer, the output is the log probability of moves
        # on each slot on the board
        self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                         units=board_height * board_width,
                                         activation=tf.nn.log_softmax)
        # 4 Evaluation Networks
        self.evaluation_conv = tf.layers.conv2d(inputs=self.block2, filters=n_filter_out,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                activation=tf.nn.relu)
        self.evaluation_conv_flat = tf.reshape(
                self.evaluation_conv, [-1, n_filter_out * board_height * board_width])
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu)
        # output the score of evaluation on current state
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1, activation=tf.nn.tanh)

        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_fc2
        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                       self.evaluation_fc2)
        # 3-2. Policy Loss function
        self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, board_height * board_width])
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file)
            print 'old model loaded!!!'

    def residual_block(self, input_tensor, n_filters):
        conv1 = tf.layers.conv2d(inputs=input_tensor, filters=n_filters,
                                 kernel_size=[3, 3], padding="same",
                                 activation=None)
        bn1 = tf.contrib.layers.batch_norm(conv1,
                                           center=True, scale=True, is_training=True)
        conv1 = tf.nn.relu(bn1)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=n_filters,
                                 kernel_size=[3, 3], padding="same",
                                 activation=None)
        bn2 = tf.contrib.layers.batch_norm(conv2,
                                           center=True, scale=True, is_training=True)
        conv22 = tf.nn.relu(bn2)
        block = input_tensor + conv22

        return block

    def residual_block_diff(self, input_tensor, n_filters):
        conv1 = tf.layers.conv2d(inputs=input_tensor, filters=n_filters,
                                 kernel_size=[3, 3], padding="same",
                                 activation=None)
        bn1 = tf.contrib.layers.batch_norm(conv1,
                                           center=True, scale=True, is_training=True)
        conv1 = tf.nn.relu(bn1)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=n_filters,
                                 kernel_size=[3, 3], padding="same",
                                 activation=None)
        bn2 = tf.contrib.layers.batch_norm(conv2,
                                           center=True, scale=True, is_training=True)
        conv22 = tf.nn.relu(bn2)

        resized_input = tf.nn.conv2d(inputs=input_tensor, filters=n_filters,
                                     kernel_size=[1, 1], padding="same",
                                     activation=None)

        block = resized_input + conv22
        return block

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probs, value = self.session.run(
                [self.action_fc, self.evaluation_fc2],
                feed_dict={self.input_states: state_batch}
                )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr})
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
