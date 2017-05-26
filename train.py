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
discount_factor = 1

# replay memory
init_replay_memory_size = 5000
replay_memory_size = 50000

batch_size = 64
freq_update_target = 5000
freq_save_model = 500
freq_print_info = 1

max_iteration = 1000000
learning_rate = 0.001

# Epsilon
initial_epsilon = 0.5
final_epsilon = 0.01
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
    samples = random.sample(replay_memory, mini_batch_size)
    states_batch, array_states_batch, action_batch, reward_batch, next_states_batch, next_array_states_batch, done_batch = map(np.array, zip(*samples))

    # calculate target Q values by target network
    # next_states_batch_array = []
    # for next_state in next_states_batch:
    #     next_states_batch_array.append(Board.get_arrayboard(next_state))

	# q_values_next = ai.estimator.predict(sess, next_states_batch_array)
 	# best_actions = np.argmax(q_values_next, axis = 1)
    best_actions = [ai.getbestdirection(sess, next_states_batch[t], 0) for t in xrange(mini_batch_size)]
    q_values_next_target = ai.target.predict(sess, next_array_states_batch)
    targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * q_values_next_target[np.arange(mini_batch_size), best_actions]

    # update the estimator network
    # states_batch_array = []
    # for s in states_batch:
    #     states_batch_array.append(Board.get_arrayboard(s))

    loss = ai.update_estimator(sess, array_states_batch, action_batch, targets_batch, lr)
    return loss

def main(_):
    global_step = 0
    epsilon = initial_epsilon

    Transition = namedtuple("Transition", ["state", "array_state", "direction", "reward", "next_state", "next_array_state", "done"])
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

        replay_memory = []
        replay_memory_large_tile = []
        game.initialize()
        
        while(len(replay_memory) < init_replay_memory_size):
            # print(len(replay_memory))
            if len(replay_memory) % (init_replay_memory_size / 100) == 0:
                print("[INFO] Populating replay memory... %3.3f%% \r" % (len(replay_memory) * 100. / init_replay_memory_size), end="")
                sys.stdout.flush()

            # choose a direction
            direction = ai_2048.getbestdirection(sess, game.state, epsilon)
            state,new_state,reward, done = game.move(direction)
            replay_memory.append(Transition(state, Board.get_arrayboard(state), direction, reward, new_state, Board.get_arrayboard(new_state), done))
            replay_memory_large_tile.append(Transition(state, Board.get_arrayboard(state), direction, reward, new_state, Board.get_arrayboard(new_state), done))

            statistic.action[direction] += 1

            if done:
                game.initialize()

        print("[INFO] Populating replay memory... %3f%% \r" % (len(replay_memory) * 100. / init_replay_memory_size), end="")
        print("\n[INFO] Populating replay memory end!")
        sys.stdout.flush()
        tstart = time.time()

        max_score = 0
        max_tile = 0
        last_iteration_score = 0
        avg_score = 0
        for iteration in xrange(max_iteration):

            game.initialize()

            while(True):
                global_step += 1
                # choose a direction
                
                direction = ai_2048.getbestdirection(sess, game.state, epsilon)
                state,new_state,reward, done = game.move(direction)

                statistic.action[direction] += 1
                epsilon -= (epsilon-final_epsilon)/explore_steps

                current_max_tile = game.get_maxtile()
                if current_max_tile > max_tile: max_tile = current_max_tile

                if is_train == True:

                    if len(replay_memory) > replay_memory_size:
                        replay_memory.pop(0)

                    if len(replay_memory_large_tile) > replay_memory_size / 10:
                        replay_memory_large_tile.pop(0)

                    replay_memory.append(Transition(state, Board.get_arrayboard(state), direction, reward, new_state, Board.get_arrayboard(new_state), done))
                    if current_max_tile >= 8: 
                    	replay_memory_large_tile.append(Transition(state, Board.get_arrayboard(state), direction, reward, new_state, Board.get_arrayboard(new_state), done))

                    loss = (update(replay_memory, batch_size, ai_2048, sess, learning_rate) + update(replay_memory_large_tile, batch_size, ai_2048, sess, learning_rate)) / 2

                    statistic.set(learning_rate, max_score,last_iteration_score, iteration+1, epsilon, global_step, loss, max_tile, game.illegal_count)

                    # update the target network
                    if global_step % freq_update_target == 0:
                    # if done:
                        ai_2048.update_target(sess)


                if done:
                    avg_score += game.score
                    if ((iteration+1) % freq_print_info) == 0:
                        print ("[%s] Iteration %d, max score: %d, max_tile: %d, score: %d, average score: %f" 
                            % ((datetime.strftime(datetime.now(), "%Y/%m/%d %H:%M:%S"), iteration+1, max_score, max_tile,game.score, avg_score*1./50,)), end="")
                        if is_train == True:
                            print(", loss: %f" % loss, end='')
                        print("")
                        avg_score = 0
                    
                    sys.stdout.flush()
                    max_score = game.score if game.score > max_score else max_score
                    last_iteration_score = game.score
                    break

            if is_debug == False:
                ai_2048.estimator.write_summary(statistic)
                if(iteration+1) % freq_save_model == 0 and is_save_model == True:
                    saver.save(sess, checkpoint_path, global_step = iteration+1)

        tend = time.time()
        print("Execution time: %f" % (tend - tstart))


if __name__ == '__main__':
    tf.app.run()