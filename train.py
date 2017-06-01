from __future__ import print_function
import tensorflow as tf
import numpy as np
from BitBoard import Board
from AI import AI
import os, sys
import time
import random
from collections import namedtuple
from datetime import datetime
from BitBoard import penalty
from Game import Game

is_debug = False
is_train = True
# Hyper parameter
discount_factor = 1.0

# replay memory
init_replay_memory_size = 8
replay_memory_size = 8

batch_size = 8
freq_update_target = 1000
freq_save_model = 500
freq_print_info = 50

max_iteration = 1000000
learning_rate = 0.00001

# Epsilon
initial_epsilon = 0.0
final_epsilon = 0.00
explore_steps = 5000000

checkpoint_path = "log/checkpoint/"
is_load_model = False
is_save_model = True

if is_debug == True:
	init_replay_memory_size = 1000
	replay_memory_size = 3000
	batch_size = 32
	freq_update_target = 100
	max_iteration = 1000
	explore_steps = 10000

Transition = namedtuple("Transition", ["states", "four_rewards", "four_next_states", "done"])

class Statistic(object):
	def __init__(self):
		self.action = [0,0,0,0]
		self.learning_rate = 0
		self.max_score = 0
		self.score = 0
		self.iteration = 0
		self.epsilon = 0
		self.global_step = 0
		self.loss = 0
		self.max_tile = 0
		self.illegal_count = 0
	
	def set(self, lr, max_score, score, iteration, epsilon, global_step, loss, max_tile, illegal_count):
		self.learning_rate = lr
		self.max_score = max_score
		self.score = score
		self.iteration = iteration
		self.epsilon = epsilon
		self.global_step = global_step
		self.loss = loss
		self.max_tile = max_tile
		self.illegal_count += illegal_count

def update(replay_memory, mini_batch_size, ai, sess, lr):
	# sample a minibatch from replay buffer. 
	# np.random.shuffle(replay_memory)
	samples = random.sample(replay_memory, mini_batch_size)
	states_batch, four_rewards_batch, four_next_states_batch, done_batch = map(np.array, zip(*samples))
	next_state_batch = four_next_states_batch.reshape((-1, 16))
	next_state_value_batch = ai.target.predict(sess, next_state_batch)
	four_next_states_value_batch = next_state_value_batch.reshape((-1, 4))
	target_value_batch = np.amax((four_rewards_batch + four_next_states_value_batch * discount_factor ), axis=1)
	target_value_batch = target_value_batch * np.invert(done_batch).astype(np.float32) - 20 * done_batch.astype(np.float32) 

	# 8 states with 8 target value
	target_value_batch = np.repeat(target_value_batch, 8)
	loss = ai.update_estimator(sess, states_batch.reshape((-1, 16)), target_value_batch, lr)
	return loss

def build_transition(pre_after_state, rewards, after_array_states, done):
	allstates = []
	allstates.append(Board.get_arrayboard(pre_after_state))
	allstates.append(Board.get_arrayboard(Board.rotate_right(pre_after_state)))
	allstates.append(Board.get_arrayboard(Board.rotate_right(Board.rotate_right(pre_after_state))))
	allstates.append(Board.get_arrayboard(Board.rotate_left(pre_after_state)))
	rev_pre_after_state = Board.mirror(pre_after_state)
	allstates.append(Board.get_arrayboard(rev_pre_after_state))
	allstates.append(Board.get_arrayboard(Board.rotate_right(rev_pre_after_state)))
	allstates.append(Board.get_arrayboard(Board.rotate_right(Board.rotate_right(rev_pre_after_state))))
	allstates.append(Board.get_arrayboard(Board.rotate_left(rev_pre_after_state)))
	allstates = np.asarray(allstates)
	return Transition(allstates, rewards, after_array_states, done)

def main(_):
	epsilon = initial_epsilon

	ai_2048 = AI()
	game = Game()
	statistic = Statistic()
	print("[INFO] Populate the replay memory")
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		
		if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)

		saver = tf.train.Saver()
		# Load a previous checkpoint if we find one
		# if is_load_model == True: saver.restore(sess, checkpoint_path)
		if is_load_model == True: saver.restore(sess, 'log/checkpoint/-16000')


		global_step = sess.run([tf.contrib.framework.get_global_step()])[0]
		
		replay_memory = []
		############################################################
		# 					Filled up replay memory				   #
		############################################################
		game.initialize()
		pre_after_state = None
		while(len(replay_memory) < init_replay_memory_size):
			# print(len(replay_memory))
			# if len(replay_memory) % (init_replay_memory_size / ) == 0:
			# 	print("[INFO] Populating replay memory... %3.3f%% \r" % (len(replay_memory) * 100. / init_replay_memory_size), end="")
			# 	sys.stdout.flush()

			# choose a direction
			direction, rewards, after_array_states = ai_2048.getbestdirection(sess, game.state, epsilon)
			state, new_state, _, done = game.move(direction)

			if pre_after_state != None:
				#small_state, _ = Board.get_feature_state(pre_after_state)

				replay_memory.append(build_transition(pre_after_state, rewards, after_array_states, done))
			
			statistic.action[direction] += 1
			if done:
				game.initialize()
				pre_after_state = None
			else:
				pre_after_state = Board.get_bitboard(after_array_states[direction])

		print("[INFO] Populating replay memory... %3f%% \r" % (len(replay_memory) * 100. / init_replay_memory_size), end="")
		print("\n[INFO] Populating replay memory end!")
		sys.stdout.flush()
		tstart = time.time()
		
		############################################################
		# 						Start training					   #
		############################################################
		global_max_score = 0
		global_max_tile = 0
		last_iteration_score = 0
		local_avg_score = 0.0
		local_avg_loss = 0.0
		local_counter = 0
		for iteration in range(max_iteration):
			game.initialize()
			pre_after_state = None
			while(True):
				global_step = global_step + 1
				# choose a direction
				
				
				if (iteration+1) % freq_print_info == 0:
					direction, rewards, after_states = ai_2048.getbestdirection(sess, game.state, 0)
				else:
					direction, rewards, after_states = ai_2048.getbestdirection(sess, game.state, epsilon)
				state, new_state, _, done = game.move(direction)

				statistic.action[direction] += 1
				if global_step < explore_steps:
					epsilon -= (epsilon-final_epsilon)/explore_steps


				if is_train == True:
					if len(replay_memory) > replay_memory_size:
						replay_memory.pop(0)
					if pre_after_state != None:
						#small_state, _ = Board.get_feature_state(pre_after_state)
						replay_memory.append(build_transition(pre_after_state, rewards, after_array_states, done))
					pre_after_state = Board.get_bitboard(after_states[direction])
					
					loss = update(replay_memory, batch_size, ai_2048, sess, learning_rate)

					statistic.set(learning_rate, global_max_score, last_iteration_score, iteration+1, epsilon, global_step, loss, global_max_tile, game.illegal_count)

					# update the target network
					if global_step % freq_update_target == 0:
						ai_2048.update_target(sess)
						
					local_counter += 1
					local_avg_loss += loss

				if done:
					local_avg_score += game.score
					global_max_tile = game.get_maxtile() if game.get_maxtile() > global_max_tile else global_max_tile
					global_max_score = game.score if game.score > global_max_score else global_max_score
					
					if ((iteration+1) % freq_print_info) == 0:
						local_avg_loss =  local_avg_loss * 1. / local_counter
						local_avg_score = local_avg_score * 1. / freq_print_info
						print ("[%.1fs] Iteration %d, global_step %d, max score: %d, max_tile: %d, score: %d, average score: %.1f" 
							% ((time.time() - tstart), iteration+1, global_step, global_max_score, global_max_tile,game.score, local_avg_score,), end="")
						if is_train == True:
							print(", average loss: %.2f" % local_avg_loss, end='')
						print("")
						local_avg_score = 0.0
						local_avg_loss = 0.0
						local_counter = 0
					sys.stdout.flush()
					last_iteration_score = game.score
					break

			if is_debug == False:
				statistic.max_score = global_max_score
				statistic.max_tile = global_max_tile
				ai_2048.estimator.write_summary(statistic)
				if(iteration+1) % freq_save_model == 0 and is_save_model == True:
					saver.save(sess, checkpoint_path, global_step = iteration+1)

		tend = time.time()
		print("Execution time: %f" % (tend - tstart))


if __name__ == '__main__':
	tf.app.run()