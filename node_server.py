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
        return sha256(block_content.encode()).hexdigest()
	
	def set_hash(self):
		# compute_hash() also used to return value for verification
		self._block_hash = self.compute_hash()
	
	def nonce_increment(self):
		self._nonce += 1

	# getters of the private attribute
	def get_block_hash(self):
		return self._block_hash
	
	def get_previous_hash(self):
		return self._previous_hash

	def get_block_idx(self):
		return self._idx

class Blockchain:

    # for PoW
    difficulty = 2

    def __init__(self):
        # it is fine to use a python list to store the chain for now
        self.chain = []

    def get_last_block(self):
        if len(self.chain) > 0:
            return self.chain[-1]
        else:
            # blockchain doesn't even have its genesis block
            return None

class Device:
	def __init__(self, idx):
		self._idx = idx
		self._data = None
		# weight dimentionality has to be the same as the data vector
        self._global_weight = None
		self._global_gradients = None
		# by default, a device is created as a worker
		self._is_miner = False
		# data dimensionality has to be predefined as an positive integer
		self._data_dim = None
		# miner can also maintain the chain, tho the paper does not mention, but we think miner can be tranferred to worked any time back and forth
		self._blockchain = Blockchain()
		# only for miner
		self._unmined_transactions = []

	# set data dimension
	def set_data_dim(self, data_dim)
		self._data_dim = data_dim

	# change role to miner by self assigning
	def set_miner(self):
		# flip a coin
		flip = random.randint(0, 1)
		if flip == 0:
			self._is_miner = True
		else:
			self._is_miner = False

	''' Functions for Workers '''

	# worker device global weight initialization or update
	def worker_set_global_weihgt(self, weight=None):
		if not self._is_miner:
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
		if not self._is_miner:
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

	''' Functions for Miners '''

	# TODO rewards
	def proof_of_work(self, candidate_block):
		''' Brute Force the nonce. May change to PoS by Dr. Jihong Park '''
		if self._is_miner:
			current_hash = candidate_block.compute_hash()
			while not current_hash.startswith('0' * Blockchain.difficulty):
				candidate_block.nonce_increment()
				current_hash = candidate_block.compute_hash()
			# return the qualified hash as a PoW proof, to be verified by other devices before adding the block
			return current_hash
		else:
			sys.exit('Worker does not perform PoW.')
	
	def receive_worker_updates(self, transaction):
		if self._is_miner:
			self._unmined_transactions.append(transaction)
		else:
			sys.exit("Worker cannot receive other workers' updates.")

	def mine_transactions(self):
		if self._is_miner:
			if self._unmined_transactions:
				
				# get the last block and construct the candidate block
				last_block = self._blockchain.get_last_block()

				candidate_block = Block(idx=last_block.get_block_idx+1,
				transactions=self._unmined_transactions,
				timestamp=time.time(),
				previous_hash=last_block.get_previous_hash())

				# mine the candidate block by PoW
				pow_proof = self.proof_of_work(candidate_block)
				#TODO broadcast the block


	''' Common Methods '''

	def add_block(self, block_to_add, pow_proof):
        """
        A function that adds the block to the chain after two verifications(sanity check).
        """
        if self.blockchain.get_last_block() is not None:
            # 1. check if the previous_hash referred in the block and the hash of latest block in the chain match.
            last_block_hash = self.get_last_block.get_block_hash()
            if block_to_add.get_previous_hash() != last_block_hash:
                # to be used as condition check later
                return False
            # 2. check if the proof is valid.
            if not check_pow_proof(block_to_add, pow_proof):
                return False
            # All verifications done.
            self.blockchain.chain.append(block_to_add)
            return True
        else:
            # add genesis block
            if not check_pow_proof(block_to_add, pow_proof):
                return False
            self.blockchain.chain.append(block_to_add)
            return True
	
	@staticmethod
	def check_pow_proof(block_to_check, pow_proof):
		# if not (block_to_add._block_hash.startswith('0' * Blockchain.difficulty) and block_to_add._block_hash == pow_proof): WRONG
		# shouldn't check the block_hash directly as it's not trustworthy and it's also private
		if pow_proof.startswith('0' * Blockchain.difficulty) and pow_proof == block_to_check.compute_hash():
			return True
		else:
			return False

	''' consensus algorithm for the longest chain '''
	
	@classmethod
	def check_chain_validity(cls, chain_to_check):
		for block in chain_to_check[1:]:
			if cls.check_chain_validity(block, block.get_block_hash()) and block.get_previous_hash == chain_to_check[chain_to_check.index(block) - 1].get_block_hash():
				pass
			else:
				return False
		return True

	# TODO
	def consensus(self):
		"""
		Simple consensus algorithm - if a longer valid chain is found, the current device's chain is replaced with it.
		"""

		longest_chain = None
		current_chain_len = len(self.blockchain.chain)

		for node in peers:
			response = requests.get('{}/chain'.format(node))
			length = response.json()['length']
			chain = response.json()['chain']
			if length > current_len and blockchain.check_chain_validity(chain):
				# Longer valid chain found!
				current_len = length
				longest_chain = chain

		if longest_chain:
			blockchain = longest_chain
			return True

		return False

	
	
''' Questions
	1. Who's in charge of assigning which devices are workers and which are the miners, if there is not a central server? Self assign?

# useful docs
''' numpy create a vector https://www.oreilly.com/library/view/machine-learning-with/9781491989371/ch01.html '''


