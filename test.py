from __future__ import print_function
import tensorflow as tf
import numpy as np
from BitBoard import Board
from AI import AI
import os, sys
import random

# Hyper parameter
max_iteration = 100
initial_epsilon = 0.01
# replay memory
checkpoint_path = "checkpoint/-21500"
is_load_model = True

def main(_):
    epsilon = initial_epsilon

    ai_2048 = AI()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        saver = tf.train.Saver()
        # Load a previous checkpoint if we find one
        if is_load_model == True: saver.restore(sess, checkpoint_path)

        state = 0x3200200000000000
        prediction1 = ai_2048.estimator.predict(sess, np.expand_dims(Board.get_arrayboard(state), 0))
        for i in range(4):
            moved_state, reward = Board.move(state,i)
            Board.printboard(moved_state)
            print("value: %f" % prediction1[0][i])
            print("------------------------------------------------")
        state2 = 0x0320020000000000
        prediction2 = ai_2048.estimator.predict(sess, np.expand_dims(Board.get_arrayboard(state2), 0))
        for i in range(4):
            moved_state2, reward2 = Board.move(state2, i)
            Board.printboard(moved_state2)
            print("value: %f" % prediction2[0][i])
            print("------------------------------------------------")

        return 0
        for iteration in xrange(max_iteration):

            state = Board.initialize()
            score = 0
            while(True):
                print("current state: ")
                Board.printboard(state)
                print("prediction: " , ai_2048.estimator.predict(sess, np.expand_dims(Board.get_arrayboard(state), 0)))
                direction = ai_2048.getbestdirection(sess, state, epsilon)
                print("direction: ", direction)
                # execute the action
                new_state,reward = Board.move(state,direction)
                print("moved state: ")
                Board.printboard(new_state)
                if reward == -1: 
                	continue
            	new_state = Board.add_random_tile(new_state)
                print("add new tile: ")
                Board.printboard(new_state)
                score += reward
                done = Board.is_end(new_state)
                state = new_state

                if done:
                    break



if __name__ == '__main__':
    tf.app.run()
