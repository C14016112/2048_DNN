import numpy as np
from BitBoard import Board
from VN import VN
import tensorflow as tf
class AI(object):

	def __init__(self):
		self.estimator = VN(scope='estimator', summaries_dir = 'log')
		self.target = VN(scope='target', summaries_dir = None)
		self.update_op = []

		variables_estimator = [t for t in tf.trainable_variables() if t.name.startswith(self.estimator.scope)]
		variables_estimator = sorted(variables_estimator, key = lambda v: v.name)
		variables_target = [t for t in tf.trainable_variables() if t.name.startswith(self.target.scope)]
		variables_target = sorted(variables_target, key = lambda v: v.name)
		for n_estimator, n_target in zip(variables_estimator, variables_target):
			op = n_target.assign(n_estimator)
			self.update_op.append(op)

	def getbestdirection(self, sess, state, epsilon):
		best_dir, rewards, next_states = self.epsilon_policy(sess, state, epsilon)
		return best_dir, rewards, next_states 

	def update_estimator(self, sess, array_state, label, learning_rate):
		loss = self.estimator.update(sess, array_state, label, learning_rate)
		return loss

	def update_target(self, sess):
		sess.run(self.update_op)

	def epsilon_policy(self, sess, state, epsilon):
		action_prob = np.ones(4, dtype = float) * epsilon / 4
		best_action = -1
		max_value = -1
		rewards = []
		next_states = []
		for i in range(4):
			tmp_state, reward = Board.move(state, i)
			rewards.append(reward)
			#small_state, _ = Board.get_feature_state(tmp_state)
			next_states.append(Board.get_arrayboard(tmp_state))
			if reward < 0:
				action_prob[i] = 0
			else:
				state_value = self.estimator.predict(sess, np.expand_dims(next_states[i], 0))[0]
				future_reward = reward + state_value
				if best_action == -1:
					best_action = i
					max_value = future_reward
				elif max_value < future_reward:
					best_action = i
					max_value = future_reward

		action_prob[best_action] += 1 - np.sum(action_prob)
		action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
		return action, np.asarray(rewards), np.asarray(next_states)