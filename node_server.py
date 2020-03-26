import sys
import random
import time
import numpy as np

import json
from hashlib import sha256


# reference - https://developer.ibm.com/technologies/blockchain/tutorials/develop-a-blockchain-application-from-scratch-in-python/#7-establish-consensus-and-decentralization
class Block:
    def __init__(self, idx, transactions, timestamp, previous_hash, nonce=0):
        self._idx = idx
        self._transactions = transactions
        self._timestamp = timestamp
        self._previous_hash = previous_hash
        self._nonce = nonce
        # the hash of the current block, calculated by compute_hash
        self._block_hash = None
    
    def compute_hash(self):
        block_content = json.dumps(self.__dict__, sort_keys=True)
        self._block_hash = sha256(block_content.encode()).hexdigest()

class Blockchain:

    # PoW
    difficulty = 2

    def __init__(self):
        # it is fine to use a python list to store the chain for now
        self._chain = []

    def last_block(self):
        if len(self._chain) > 0:
            return self._chain[-1]
        else:
            # blockchain doesn't even have its genesis block
            return None
    
    def add_block(self, block_to_add, pow_proof):
        """
        A function that adds the block to the chain after two verifications.
        """
        
        if self.last_block() is not None:
            # 1. check if the previous_hash referred in the block and the hash of latest block in the chain match.
            last_block_hash = self._chain[-1]._block_hash
            if block_to_add._block_hash != last_block_hash:
                # to be used as condition check later
                return False
            # 2. check if the proof is valid.
            if not (block_to_add._block_hash.startswith('0' * Blockchain.difficulty) and block_to_add._block_hash == pow_proof):
                return False
            # All verifications done.
            self._chain.append(block_to_add)
            return True
        else:
            # add genesis block
            if not (block_to_add._block_hash.startswith('0' * Blockchain.difficulty) and block_to_add._block_hash == pow_proof):
                return False
            self._chain.append(block_to_add)
            return True

class Device:
	def __init__(self, idx):
		self._idx = idx
		self._data = None
		# weight dimentionality has to be the same as the data vector
        self._global_weight = None
		self._global_gradients = None
		# by default, a device is created as a worker
		self._miner = False
		# data dimensionality has to be predefined as an positive integer
		self._data_dim = None

	# set data dimension
	def set_data_dim(self, data_dim)
		self._data_dim = data_dim

	# change role to miner
	def set_miner(self):
		self._miner = True

	''' Functions for Workers '''

	# worker device global weight initialization or update
	def worker_set_global_weihgt(self, weight=None):
		if not self._miner:
			if not weight:
				# if not updating, initialize with all 0s
				# TODO Need to ask authors what's the initial states for all the devices. Shall they have the same global weights, or can be different
				self._global_weight = np.zeros((self._data_dim, 1))
				# for _ in self._data_dim:
				# 	self._global_weight.append(random.uniform(0, 1))
			else:
				self._global_weight = weight
		else:
			sys.exit("Miner does not set weight values")

	# worker data sample initialization
	def worker_set_data_samples(self, data):
		# data is a collection of data points, and each data point {x, y} has a len(self._data_dim)-dim column vector x and a scalar value y. For simplicity, each data point is represented as a tuple, e.g. data = [(a, 3), (b, 4)...] where a and b are column vectors with len(self._data_dim)-dim in mumpy format
		if not self._miner:
			if data[0][0].shape[0] != self._global_weight.shape[0]:
				sys.exit("feature dimentionality has to be the same with the weight vector")
			self._data = data
		else:
			sys.exit("Miner does not initilize data points")
	
	# BlockFL step 1 - train with regression
	# return local computation time, and delta_fk(wl) as a list
	# global_gradient is calculated after updating the global_weights
	def worker_local_update(self, step_size):
		# SVRG algo, BlockFL section II and reference[4] 3.2
		# gradient of loss function chosen - mean squared error
		# delta_fk(wl)
		gradients_global_per_data_point = []
		# initialize local weights as the (recalculated) global weights
		local_weight = self._global_weight
		# ref - https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module
		start_time = time.time()
		# iterations = the number of data points in a device
		for data_point in self._data:
			local_weight = local_weight - (step_size/len(self._data)) * (data_point[0].transpose()@local_weight - data_point[1])

			self._global_weight[data_iter] = self._global_weight - (step_size/iterations) * ()
		return time.time() - start_time


# useful docs
''' numpy create a vector https://www.oreilly.com/library/view/machine-learning-with/9781491989371/ch01.html '''


