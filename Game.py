from __future__ import print_function
from BitBoard import Board
import numpy as np

class Game(object):

	def __init__(self):
		self.state = 0x0
		self.previous_state = 0x0
		self.board = Board
		self.add_random_tile()
		self.add_random_tile()
		self.score = 0
		self.illegal_count = 0

	def initialize(self):
		self.state = 0x0
		self.previous_state = 0x0
		self.add_random_tile()
		self.add_random_tile()
		self.score = 0
		self.illegal_count = 0

	def is_teminate(self):
		for i in xrange(4):
			state, reward = self.board.move(self.state,i)
			if reward >= 0:
				return False
		return True

	def add_random_tile(self): 
		space = np.zeros([16])
		space[:] = 0x0
		num = 0
		for i in xrange(16):
			if(self.board.at(self.state,i) == 0):
				space[num] = i
				num += 1
		if (num > 0):
			self.state = self.board.set(self.state, np.int(space[np.random.randint(100) % num]), 2 if np.random.randint(4) == 0 else 1 )

	# def game_start(self):
	def move(self, direction):
		self.previous_state = self.state
		self.state, reward = Board.move(self.state, direction)
		done = self.is_teminate()
		self.illegal_count = 1 if reward < 0 else 0
		self.score += reward
		self.add_random_tile()
		return self.previous_state, self.state, reward, done
		
	def printboard(self):
		print("-------------------------------------")
		for i in xrange(4):
			for j in xrange(4):
				print(" %5d " % self.board.at(self.state,i*4+j) , end = "")
			print("")
		print("-------------------------------------")

	def get_maxtile(self):
		max_tile = 0
		for i in xrange(16):
			tile = self.board.at(self.state,i)
			if tile > max_tile: max_tile = tile
		return max_tile

