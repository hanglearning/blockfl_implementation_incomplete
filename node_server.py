import sys
import random
import time
import torch
import os
import binascii

import json
from hashlib import sha256


# reference - https://developer.ibm.com/technologies/blockchain/tutorials/develop-a-blockchain-application-from-scratch-in-python/
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

	def get_transactions(self):
		# get the updates from this block
		return self._transactions

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
		# by default, a device is created as a worker
		self._is_miner = False
		''' attributes for workers '''
		# data is a python list of samples, within which each data sample is a dictionary of {x, y}, where x is a numpy column vector and y is a scalar value
		self._data = []
		# weight dimentionality has to be the same as the dim of the data column vector
        self._global_weight_vector = None
		# self._global_gradients = None
		self._step_size = None
		# data dimensionality has to be predefined as an positive integer
		self._data_dim = None
		# sample size(Ni)
		self._sample_size = None
		# miner can also maintain the chain, tho the paper does not mention, but we think miner can be tranferred to worked any time back and forth
		self._blockchain = Blockchain()
		''' attributes for miners '''
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

	def is_miner(self):
		return self.is_miner


	''' Functions for Workers '''

	def worker_set_sample_size(self, sample_size):
		if self._is_miner:
			sys.exit("Sample size is not required for miners.")
		else:
			self._sample_size = sample_size

	def worker_set_step_size(self, step_size):
		if self._is_miner:
			sys.exit("Step size is only for workers to calculate weight updates.")
		else:
			if step_size <= 0:
				sys.exit("Step size has to be positive.")
			else:
				self._step_size = step_size

	def worker_generate_dummy_data(self):
		# https://stackoverflow.com/questions/15451958/simple-way-to-create-matrix-of-random-numbers
		if self._is_miner:
			sys.exit("Miner does not own data samples to update.")
		else:
			if not self._data:
				for _ in range(self._sample_size):
					self._data.append({'x': torch.randint(0, high=20, size=(self._data_dim, 1)), 'y': torch.randint(0, high=20, size=(1, 1))})
			else:
				sys.exit("The data of this worker has already been initialized.")

	# worker global weight initialization or update
	def worker_set_global_weihgt(self, weight=None):
		if self._is_miner:
			sys.exit("Miner does not set weight values")
		else:
			if not self._global_weight_vector:
				# if not updating, initialize with all 0s, as directed by Dr. Park
				# Or, we should hard code a vector with some small values for the device class as it has to be the same for every device
				self._global_weight_vector = torch.zeros(self._data_dim, 1)
			else:
				self._global_weight_vector = weight
			
	
	# BlockFL step 1 - train with regression
	# return local computation time, and delta_fk(wl) as a list
	# global_gradient is calculated after updating the global_weights
	def worker_local_update(self):
		if self._is_miner:
			sys.exit("Miner does not perfrom gradient calculations.")
		else:
			# SVRG algo, BlockFL section II and reference[4] 3.2
			# gradient of loss function chosen - mean squared error
			# delta_fk(wl)
			global_gradients_per_data_point = []
			# initialize the local weights as the current global weights
			local_weight = self._global_weight_vector
			# calculate delta_f(wl)
			last_block = self._blockchain.get_last_block()
			if last_block is not None:
				transactions = last_block.get_transactions()
				# transactions = [{'updated_weigts': w, 'updated_gradients': [f1wl, f2wl ... fnwl]} ... ]
				tensor_accumulator = torch.zeros_like(self._global_weight_vector)
				for update_per_device in transactions:
					for data_point_gradient in update_per_device['updated_gradients']:
						tensor_accumulator += data_point_gradient
				num_of_device_updates = len(transactions)
				delta_f_wl = tensor_accumulator/(num_of_device_updates * self._sample_size)
			else:
				# chain is empty now as this is the first epoch. To keep it consistent, we set delta_f_wl as 0 tensors
				delta_f_wl = torch.zeros_like(self._global_weight_vector)
			# ref - https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module
			start_time = time.time()
			# iterations = the number of data points in a device
			# TODO function(1)
			for data_point in self._data:

				local_weight_track_grad = torch.tensor(local_weight, requires_grad=True)
				# loss of one data point with current local update fk_wil
				fk_wil = (data_point['x'].t()@local_weight_track_grad - data_point['y'])**2/2
				# calculate delta_fk_wil
				fk_wil.backward()
				delta_fk_wil = local_weight_track_grad.grad

				last_global_weight_track_grad = torch.tensor(self._global_weight_vector, requires_grad=True)
				# loss of one data point with last updated global weights fk_wl
				fk_wl = (data_point['x'].t()@last_global_weight_track_grad - data_point['y'])**2/2
				# calculate delta_fk_wl
				fk_wl.backward()
				delta_fk_wl = last_global_weight_track_grad.grad
				# record this value to update
				global_gradients_per_data_point.append(delta_fk_wl)

				# calculate local update
				local_weight = local_weight - (step_size/len(self._data)) * (delta_fk_wil - delta_fk_wl+ delta_f_wl)

			return local_weight, global_gradients_per_data_point， time.time() - start_time

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

app = Flask(__name__)

# pre-defined and agreed fields
data_dim = 10
sample_size = 20
step_size = 3

# create a device with a 4 bytes (8 hex chars) id
device = Device(binascii.b2a_hex(os.urandom(4)).decode('utf-8'))
# set data dimension
set_data_dim(data_dim)
# the device's copy of blockchain
blockchain = Blockchain()

# start the app
# assign tasks based on role
@app.route('/')
def runApp():
	# assign/change role of this device
	device.set_miner()
	if device.is_miner():
		
	else:
		device.worker_set_sample_size(sample_size)
		device.worker_set_step_size(3)
		# TODO first do consensus to get the longest blockchain from peers
		device.worker_generate_dummy_data()



# the address to other participating members of the network
peers = set()


	
''' Questions
	1. Who's in charge of assigning which devices are workers and which are the miners, if there is not a central server? Self assign?
'''

# useful docs
''' 
pytorch create a vector https://pytorch.org/docs/stable/torch.html 

seems int operations are faster than float, so use column vector that are full of ints http://nicolas.limare.net/pro/notes/2014/12/12_arit_speed/

create torch tensor from np array https://pytorch.org/docs/stable/tensors.html

use pytorch autograd to calculate gradients
https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
'''

