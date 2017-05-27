import numpy as np
from BitBoard import Board
from DQN import DQN
import tensorflow as tf
class AI(object):

	def __init__(self):
		self.estimator = DQN(scope='estimator', summaries_dir = 'log')
		self.target = DQN(scope='target', summaries_dir = None)
		self.update_op = []

		variables_estimator = [t for t in tf.trainable_variables() if t.name.startswith(self.estimator.scope)]
		variables_estimator = sorted(variables_estimator, key = lambda v: v.name)
		variables_target = [t for t in tf.trainable_variables() if t.name.startswith(self.target.scope)]
		variables_target = sorted(variables_target, key = lambda v: v.name)
		for n_estimator, n_target in zip(variables_estimator, variables_target):
			op = n_target.assign(n_estimator)
			self.update_op.append(op)

	def getbestdirection(self, sess, state, epsilon):
		# feature_state, op_id = Board.get_feature_state(state)
		# array_state = Board.get_arrayboard(feature_state)
		# best_dir = self.epsilon_policy(sess, array_state, epsilon)
		# best_dir = Board.operation_id_to_action(op_id, best_dir)
		best_dir = self.epsilon_policy(sess, state, epsilon)
		return best_dir

	def update_estimator(self, sess, state, action, label, learning_rate):
		loss = self.estimator.update(sess, state, action, label, learning_rate)
		return loss

	def update_target(self, sess):
		sess.run(self.update_op)

	def epsilon_policy(self, sess, state, epsilon):
		best_dir = 0
		best_val = -1
		for i in range(4):
			new_state, reward = Board.move(state, i)
			if new_state == state: continue
			value = self.estimator.predict(sess, np.expand_dims(Board.get_arrayboard(new_state), 0))[0]
			if value + reward > best_val:
				best_val = value + reward
				best_dir = i
			# print(i, " ", value + reward)
		return best_dir

