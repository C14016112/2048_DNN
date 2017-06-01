import tensorflow as tf
import os, sys
import numpy as np
from BitBoard import Board
class VN(object):

	def __init__(self, scope = "VN", summaries_dir=None):
		self.scope = scope
		self.summary_writer = None
		self.use_bn = False
		self.use_dropout = False
		self.keep_prob = 0.8
		with tf.variable_scope(scope):
			self.build_model()
		if summaries_dir:
			summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
			if not os.path.exists(summary_dir):
				os.makedirs(summary_dir)
			self.summary_writer = tf.summary.FileWriter(summary_dir)

	def fully_connected(self, input, width, scope, activation_fn = tf.nn.relu, use_bn = True, use_dropout=True):
		result = tf.contrib.layers.fully_connected(input, width, scope=scope, activation_fn=activation_fn)
		result = tf.layers.batch_normalization(result) if (use_bn&self.use_bn) == True else result
		result = tf.nn.dropout(result, self.keep_prob) if (use_dropout&self.use_dropout) == True else result
		return result

	def build_model(self):
		self.lr = tf.placeholder(shape=[], dtype=tf.float32, name="learning_rate")
		self.X = tf.placeholder(shape = [None, 256], dtype = tf.int32, name = 'X')
		self.Y = tf.placeholder(shape = [None], dtype=tf.float32, name='Y')
		X = tf.to_float(self.X) # modify
		#X = tf.one_hot(self.X , depth=16, dtype=tf.float32)
		#X = tf.reshape(X, shape = [-1, 256])
		#embeddings = tf.Variable(tf.random_uniform([16, 16], -1.0, 1.0))
		#X = tf.reshape(self.X , shape = [-1])
		#embed_X = tf.nn.embedding_lookup(embeddings, X)
		#embed_X = tf.reshape(embed_X , shape = [-1, 256])
		layer = []
		layer2 = []
		layer3 = []
		layer4 = []
		for i in xrange(4):
			layer.append(self.fully_connected(X[:,i*64:(i+1)*64-1], 256, 'layer1_%d' % i, tf.nn.relu))
			layer2.append(self.fully_connected(layer[i], 256, 'layer2_%d' % i, tf.nn.relu))
			layer3.append(self.fully_connected(layer2[i], 128, 'layer3_%d' % i, tf.nn.relu))
			layer4.append(self.fully_connected(layer3[i], 128, 'layer4_%d' % i, tf.nn.relu))

		concat = tf.concat([l for l in layer4],1)
		layer5 = self.fully_connected(concat, 512, 'layer5', tf.nn.relu)
		layer6 = self.fully_connected(concat, 512, 'layer6', tf.nn.relu)
		layer7 = self.fully_connected(concat, 256, 'layer7', tf.nn.relu)
		layer8 = self.fully_connected(concat, 256, 'layer8', None)
		self.predictions = self.fully_connected(layer8, 1, 'prediction', None, False, False)

		self.losses = tf.squared_difference(self.Y, self.predictions)
		self.loss = tf.reduce_mean(self.losses)
		# self.op = tf.train.RMSPropOptimizer(self.lr)
		self.op = tf.train.AdamOptimizer(self.lr)
		self.train_step = self.op.minimize(self.loss, global_step=tf.contrib.framework.get_or_create_global_step())
		
		
		self.summaries = tf.summary.merge([
				tf.summary.scalar("loss", self.loss),
				#tf.summary.histogram("loss_hist", self.losses),
				tf.summary.histogram("values_hist", self.predictions),
				tf.summary.histogram("target_values_hist", self.Y),
				#tf.summary.histogram("tile_hist", self.X),
				tf.summary.scalar("max_value", tf.reduce_max(self.predictions))
			])

	def predict(self, sess, state):
		return sess.run(self.predictions, {self.X:self.to_one_hot_input(state)})

	def update(self, sess, state, label, learning_rate):
		feed_dict = {
			self.X : self.to_one_hot_input(state),
			self.Y : label,
			self.lr : learning_rate
		}
		_, loss, summary, step = sess.run([self.train_step, self.loss, self.summaries, tf.contrib.framework.get_global_step()], feed_dict)
		if self.summary_writer:
			self.summary_writer.add_summary(summary, step)
		return loss

	def write_summary(self, statistic):
		# Summaries for Tensorboard
		if self.summary_writer:
			with tf.name_scope('summaries'):
				summary = tf.Summary()
				summary.value.add(tag="score", simple_value=statistic.score)
				#summary.value.add(tag='loss', simple_value=statistic.loss)
				#summary.value.add(tag='learning rate', simple_value=statistic.learning_rate)
				summary.value.add(tag='max score', simple_value=statistic.max_score)
				summary.value.add(tag='epsilon', simple_value=statistic.epsilon)
				#summary.value.add(tag='global_step', simple_value=statistic.global_step)
				summary.value.add(tag='max tile', simple_value=statistic.max_tile)
				summary.value.add(tag='illegal count', simple_value = statistic.illegal_count)
			self.summary_writer.add_summary(summary, statistic.iteration)

	def to_one_hot_input(self, states):
		'''one_hot_input = []
		print(states[0])
		for s in states:
			one_hot_each_state = np.zeros(256)
			for i in range(16):
				one_hot_each_state[i*16+s[i]] = 1 
			one_hot_input.append(one_hot_each_state)
		print(one_hot_input[0])'''

		one_hot_input = [] 
		for s in states:
			one_hot_each_state = np.zeros(shape=(16, 4, 4))
			for i in range(4):
				for j in range(4):
					one_hot_each_state[s[i * 4 + j], i, j] = 1
			one_hot_each_state = one_hot_each_state.reshape([256])
			one_hot_input.append(one_hot_each_state)
		return np.asarray(one_hot_input)

