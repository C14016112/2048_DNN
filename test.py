from __future__ import print_function
import tensorflow as tf
import numpy as np
from BitBoard import Board
from AI import AI
import os, sys
import random
from Game import Game
# Hyper parameter
max_iteration = 10
initial_epsilon = 0.01
# replay memory
checkpoint_path = "log/checkpoint/-16000"
is_load_model = True

def main(_):
    epsilon = initial_epsilon

    ai_2048 = AI()
    game = Game()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        saver = tf.train.Saver()
        # Load a previous checkpoint if we find one
        if is_load_model == True: saver.restore(sess, checkpoint_path)

        for iteration in xrange(max_iteration):
            game.initialize()
            while(True):
                print("current state: ")
                game.printboard()
                #print("prediction: " , ai_2048.estimator.predict(sess, np.expand_dims(Board.get_arrayboard(game.state), 0)))
                direction = ai_2048.getbestdirection(sess, game.state, epsilon)
                print("direction: ", direction)
                # execute the action
                state,new_state,reward, done = game.move(direction)
                if done:
                    break



if __name__ == '__main__':
    tf.app.run()
