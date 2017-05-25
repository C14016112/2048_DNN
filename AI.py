import numpy as np
from BitBoard import Board
from DQN import DQN
import tensorflow as tf
class AI(object):

	def __init__(self):
		self.estimator = DQN(scope='estimator', summaries_dir = 'summaries_estimator')
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
		best_dir = self.epsilon_policy(sess, state, Board.get_arrayboard(state), epsilon)
		return best_dir

	def update_estimator(self, sess, state, action, label, learning_rate):
		loss = self.estimator.update(sess, state, action, label, learning_rate)
		return loss

	def update_target(self, sess):
		sess.run(self.update_op)

	def epsilon_policy(self, sess, state, array_state, epsilon):
		action_prob = np.ones(4, dtype = float) * epsilon / 4
		q_values = self.estimator.predict(sess, np.expand_dims(array_state, 0))[0]
		for i in range(4):
			tmp_state, reward = Board.move(state, i)
			q_values[i] += reward
		best_action = np.argmax(q_values)
		action_prob[best_action] += (1.0 - epsilon)
		action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
		return action