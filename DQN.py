import tensorflow as tf
import os, sys
import numpy as np
from BitBoard import Board
class DQN(object):

	def __init__(self, scope = "DQN", summaries_dir=None):
		self.direction = ['up', 'right', 'down', 'left']
		self.scope = scope
		self.summary_writer = None
		self.summary_action_writer = []
		self.is_bn = False
		self.is_dropout = False
		self.keep_prob = 0.8
		self.count = 0
		with tf.variable_scope(scope):
			self.build_model()
			if summaries_dir:
				summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
				if not os.path.exists(summary_dir):
					os.makedirs(summary_dir)
				self.summary_writer = tf.summary.FileWriter(summary_dir)

				for i in xrange(4):
					action_dir = summary_dir + "/" + self.direction[i]
					if not os.path.exists(action_dir):
						os.makedirs(action_dir)
					self.summary_action_writer.append(tf.summary.FileWriter(action_dir))
		sys.stdout.flush()

	def fully_connected(self, input, width, scope, activation_fn = tf.nn.relu, is_bn = True, is_dropout=True):
		result = tf.contrib.layers.fully_connected(input, width, scope=scope, activation_fn=activation_fn)
		result = tf.layers.batch_normalization(result) if (is_bn&self.is_bn) == True else result
		result = tf.nn.dropout(result, self.keep_prob) if (is_dropout&self.is_dropout) == True else result
		return result

	def build_model(self):
		self.lr = tf.placeholder(shape=[], dtype=tf.float32, name="learning_rate")
		self.X = tf.placeholder(shape = [None,256], dtype = tf.uint8, name = 'X')
		# the target q value predicted by target network
		self.Y = tf.placeholder(shape = [None], dtype=tf.float32, name='Y')
		# self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')

		X = tf.to_float(self.X) / 15.
		batch_size = tf.shape(self.X)[0]

		self.layer1 = self.fully_connected(X, 1024, 'layer1', tf.nn.relu)
		self.layer2 = self.fully_connected(self.layer1, 512, 'layer2', tf.nn.relu)
		self.layer3 = self.fully_connected(self.layer2, 256, 'layer3', tf.nn.relu)
		# self.layer4 = self.fully_connected(self.layer3, 256, 'layer4', tf.nn.relu)
		# self.layer5 = self.fully_connected(self.layer4, 256, 'layer5', tf.nn.relu)
		self.predictions = self.fully_connected(self.layer3, 1, 'prediction', None, False, False)

		# self.gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions
		# self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), self.gather_indices)
		self.losses = tf.squared_difference(self.Y, self.predictions)
		self.loss = tf.reduce_mean(self.losses)
		self.op = tf.train.RMSPropOptimizer(self.lr)
		# self.op = tf.train.AdamOptimizer(self.lr)
		self.train_step = self.op.minimize(self.loss)

	def predict(self, sess, state):
		one_hot_state = self.to_one_hot_input(state)
		# print("layer1: ", sess.run(self.layer1, {self.X:one_hot_state}))
		# print("layer2: ", sess.run(self.layer2, {self.X:one_hot_state}))
		# print("layer3: ", sess.run(self.layer3, {self.X:one_hot_state}))
		# print("count: " , self.count, "prediction: ", sess.run(self.predictions, {self.X:one_hot_state})[0], np.argmax(sess.run(self.predictions, {self.X:one_hot_state})[0]))
		# self.count += 1
		return sess.run(self.predictions, {self.X:one_hot_state})

	def update(self, sess, state, action, label, learning_rate):
		one_hot_state = self.to_one_hot_input(state)
		feed_dict = {
			self.X : one_hot_state,
			self.Y : label,
			# self.actions : action,
			self.lr : learning_rate
		}
		loss, _, predictions = sess.run([self.loss, self.train_step, self.predictions], feed_dict)
		return loss

	def write_summary(self, statistic):
		# Summaries for Tensorboard
		if self.summary_writer:
			summary = tf.Summary()
			summary.value.add(tag="score", simple_value=statistic.score)
			summary.value.add(tag='loss', simple_value=statistic.loss)
			summary.value.add(tag='learning rate', simple_value=statistic.learning_rate)
			summary.value.add(tag='max score', simple_value=statistic.max_score)
			summary.value.add(tag='epsilon', simple_value=statistic.epsilon)
			summary.value.add(tag='global_step', simple_value=statistic.global_step)
			summary.value.add(tag='max_tile', simple_value=statistic.max_tile)
			summary.value.add(tag='illegal count', simple_value = statistic.illegal_count)
			self.summary_writer.add_summary(summary, statistic.iteration)

			for i in xrange(4):
				tmp_summary = tf.Summary(value=[tf.Summary.Value(tag='action', simple_value=statistic.action[i])])
				self.summary_action_writer[i].add_summary(tmp_summary, statistic.iteration)

	def to_one_hot_input(self, state):
		one_hot_input = []
		for s in state:
			one_hot_each_state = np.zeros(256)
			for i in range(16):
				one_hot_each_state[i*16+s[i]] = 1 
			one_hot_input.append(one_hot_each_state)
		return one_hot_input

