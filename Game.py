from BitBoard import Board
from AI import AI
import numpy as np

class Game(object):

	def __init__(self):
		self.state = 0x0
		self.board = Board
		self.state = self.board.add_random_tile(self.state)
		self.state = self.board.add_random_tile(self.state)
		self.score = 0


	def is_teminate(self):
		return self.board.is_end(self.state)

	def game_start(self):
		while(True):
			# best_direction = self.ai.getbestdirection(self.board)
			game.printboard()
			best_direction = 0
			best_direction = input()
			self.state,reward = self.board.move(self.state,best_direction)
			if reward == -1:
				print("[WARNING] The direction %d cannot move." % best_direction)
				return -1
			self.score += reward
			self.state = self.board.add_random_tile(self.state)
			if self.is_teminate() == True:
				break

	def printboard(self):
		self.board.printboard(self.state)

# for i in range(100):
game = Game()
game.game_start()
# game.printboard()