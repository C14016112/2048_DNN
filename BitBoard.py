from __future__ import print_function
import numpy as np
import os, sys
import time
'''
------------
15 14 13 12
------------
11 10  9  8
------------ 
 7  6  5  4
------------
 3  2  1  0
------------
'''
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
penalty = -20
class MoveTable(object):

	def __init__(self):
		# self.movetable = np.zeros([1<<16], dtype = np.int64)
		self.movetable = np.zeros([1<<16])
		self.movetable[:] = 0x0
		self.scoretable = np.zeros(1<<16, dtype = np.int64)

		for r in xrange(1<<16):
			V = [ (r >> 0) & 0x0f, (r >> 4) & 0x0f, (r >> 8) & 0x0f, (r >> 12) & 0x0f ]
			L = [ V[0], V[1], V[2], V[3]]
			score = self.mvleft(L)
			self.movetable[r] = ((L[0] << 0) | (L[1] << 4) | (L[2] << 8) | (L[3] << 12))
			self.scoretable[r] = score

	def mvleft(self, row):
		top = 0
		tmp = 0
		score = 0
		for i in xrange(4):
			tile = row[i]
			if tile == 0:
				continue
			row[i] = 0
			if tmp != 0:
				if tile == tmp:
					tile += 1
					row[top] = tile
					top += 1
					score += (1 << tile)
					tmp = 0
				else:
					row[top] = tmp
					top+=1
					tmp = tile
			else:
				tmp = tile
		if tmp != 0:
			row[top] = tmp
		return score

movetable = MoveTable()
class Board(object):

	@staticmethod
	def at(state, i):
		state = int(state)
		return ((state >> (i<<2)) & 0xf)

	@staticmethod
	def set(state, i, value):
		# set a component
		state = int(state)
		state = (state & ~(0x0f << (i << 2))) | ((value & 0x0f) << (i << 2))
		return state

	@staticmethod
	def fetch(state, i):
		# fetch a row
		state = int(state)
		return state >> (i << 4) & 0xffff

	@staticmethod
	def place(state, i, row):
		# set a row
		state = int(state)
		state = (state & ~(0xffff << (i << 4))) | ((row & 0xffff) << (i << 4))
		return state

	@staticmethod
	def move(state, direction):
		state = int(state)
		score = 0
		if direction == UP:
			state,score = Board.moveup(state)
		elif direction == RIGHT:
			state,score = Board.moveright(state)
		elif direction == DOWN:
			state,score = Board.movedown(state)
		else:
			state,score = Board.moveleft(state)
		return state, score

	@staticmethod
	def moveup(state):
		state = int(state)
		state = Board.rotate_left(state)
		state,score = Board.moveleft(state)
		state = Board.rotate_right(state)
		return state,score

	@staticmethod
	def moveright(state):
		state = int(state)
		state = Board.mirror(state)
		state,score = Board.moveleft(state)
		state = Board.mirror(state)
		return state,score

	@staticmethod
	def movedown(state):
		state = int(state)
		state = Board.rotate_right(state)
		state,score = Board.moveleft(state)
		state = Board.rotate_left(state)
		return state,score

	@staticmethod
	def moveleft(state):
		state = int(state)
		score = movetable.scoretable[Board.fetch(state,0)] + movetable.scoretable[Board.fetch(state,1)] + movetable.scoretable[Board.fetch(state,2)] + movetable.scoretable[Board.fetch(state,3)]
		new_state = 0x0
		for i in xrange(4):
			new_state |= np.int(movetable.movetable[Board.fetch(state,i)] * (1 << (16*i)))
		if(new_state == state):
			return state,penalty
		else:
			return new_state,score

	# @staticmethod
	# def printboard(state):
		# state = int(state)
		# print("-------------------------------------")
		# for i in xrange(4):
		# 	for j in xrange(4):
		# 		print(" %5d " % Board.at(state,i*4+j) , end = "")
		# 	print("")
		# print("-------------------------------------")

	@staticmethod
	def transpose(state):
		state = int(state)
		state = (state & 0xf0f00f0ff0f00f0f) | ((state & 0x0000f0f00000f0f0) << 12) | ((state & 0x0f0f00000f0f0000) >> 12);
		state = (state & 0xff00ff0000ff00ff) | ((state & 0x00000000ff00ff00) << 24) | ((state & 0x00ff00ff00000000) >> 24);
		return state

	@staticmethod
	def mirror(state):
		state = int(state)
		state = ((state & 0x000f000f000f000f) << 12) | ((state & 0x00f000f000f000f0) << 4) | ((state & 0x0f000f000f000f00) >> 4) | ((state & 0xf000f000f000f000) >> 12);
		return state

	@staticmethod
	def	flip(state):
		state = int(state)
		state = ((state & 0x000000000000ffff) << 48) | ((state & 0x00000000ffff0000) << 16) | ((state & 0x0000ffff00000000) >> 16) | ((state & 0xffff000000000000) >> 48);
		return state

	@staticmethod
	def rotate_right(state):
		state = int(state)
		state = Board.transpose(state)
		state = Board.mirror(state)
		return state

	@staticmethod
	def rotate_left(state):
		state = int(state)
		state = Board.transpose(state)
		state = Board.flip(state)
		return state

	@staticmethod
	def reverse(state):
		state = int(state)
		state = Board.mirror(state)
		state = Board.flip(state)
		return state

	# @staticmethod
	# def add_random_tile(state): 
	# 	state = int(state)
	# 	space = np.zeros([16])
	# 	space[:] = 0x0
	# 	num = 0
	# 	for i in xrange(16):
	# 		if(Board.at(state,i) == 0):
	# 			space[num] = i
	# 			num += 1
	# 	if (num > 0):
	# 		state = Board.set(state, np.int(space[np.random.randint(100) % num]), 2 if np.random.randint(4) == 0 else 1 )
	# 	return state
		
	# @staticmethod
	# def is_end(state):
	# 	state = int(state)
	# 	old_state = state
	# 	for i in xrange(4):
	# 		state, reward = Board.move(state,i)
	# 		if reward != penalty:
	# 			return False
	# 	return True

	# @staticmethod
	# def initialize():
	# 	state = 0x0
	# 	state = Board.add_random_tile(state)
	# 	state = Board.add_random_tile(state)
	# 	return state

	@staticmethod
	def get_arrayboard(state):
		state = int(state)
		array_state = []
		for i in xrange(16):
			array_state.append(Board.at(state,i))
		return array_state

	@staticmethod
	def get_feature_state(state):
		# [origin, rotate right 1, rotate right 2, rotate left 1, mirror, flip, transpose, reverse]
		# [  0   ,       1       ,        2      ,       3       ,   4   ,   5 ,     6    ,    7   ]
		state = int(state)
		smallest_state = state
		operation_id = 0

		new_state = Board.rotate_right(state)
		if new_state < smallest_state:
			smallest_state = new_state
			operation_id = 1
		new_state = Board.rotate_right(Board.rotate_right(state))
		if new_state < smallest_state:
			smallest_state = new_state
			operation_id = 2
		new_state = Board.rotate_left(state)
		if new_state < smallest_state:
			smallest_state = new_state
			operation_id = 3
		new_state = Board.mirror(state)
		if new_state < smallest_state:
			smallest_state = new_state
			operation_id = 4
		new_state = Board.flip(state)
		if new_state < smallest_state:
			smallest_state = new_state
			operation_id = 5
		new_state = Board.transpose(state)
		if new_state < smallest_state:
			smallest_state = new_state
			operation_id = 6
		new_state = Board.reverse(state)
		if new_state < smallest_state:
			smallest_state = new_state
			operation_id = 7
		return smallest_state,operation_id

	@staticmethod
	def bitboard_operation(state, operation_id):
		# [origin, rotate right 1, rotate right 2, rotate right 3, mirror, flip, transpose, reverse]
		# [  0   ,       1       ,        2      ,       3       ,   4   ,   5 ,     6    ,    7   ]
		if operation_id == 0:
			return state
		elif operation_id == 1:
			return Board.rotate_right(state)
		elif operation_id == 2:
			return Board.rotate_right(Board.rotate_right(state))
		elif operation_id == 3:
			return Board.rotate_left(state)
		elif operation_id == 4:
			return Board.mirror(state)
		elif operation_id == 5:
			return Board.flip(state)
		elif operation_id == 6:
			return Board.transpose(state)
		elif operation_id == 7:
			return Board.reverse(state)
		else:
			print("[ERROR] No operation id %d" % operation_id)
			return state

	@staticmethod
	def operation_id_to_action(operation_id, action):
		# [origin, rotate right 1, rotate right 2, rotate right 3, mirror, flip, transpose, reverse]
		# [  0   ,       1       ,        2      ,       3       ,   4   ,   5 ,     6    ,    7   ]
		if operation_id < 4:
			return (action+4-operation_id) % 4
		elif operation_id == 4:
			if action == 1:
				return 3
			elif action == 3:
				return 1 
			else:
				return action
		elif operation_id == 5:
			if action == 0:
				return 2
			elif action == 2:
				return 0
			else:
				return action		
		elif operation_id == 6:
			if action == 0:
				return 3
			elif action == 1:
				return 2
			elif action == 2:
				return 1
			elif action == 3:
				return 0
		elif operation_id == 7:
			if action == 0:
				return 2
			elif action == 1:
				return 3
			elif action == 2:
				return 0
			elif action == 3:
				return 1
		else:
			print("[ERROR] No operation id %d" % operation_id)
			return 0
	# @staticmethod
	# def get_maxtile(state):
	# 	state = int(state)
	# 	max_tile = 0
	# 	for i in xrange(16):
	# 		tile = Board.at(state,i)
	# 		if tile > max_tile: max_tile = tile
	# 	return max_tile