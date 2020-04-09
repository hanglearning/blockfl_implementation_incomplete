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
		# technically this should be _chain as well
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

	# get device id
	def get_idx(self):
		return self._idx

	# get device's copy of blockchain
	def get_blockchain(self):
		return self._blockchain

	# get global_weight_vector, used while being the register_with node to sync with the registerer node
	def get_global_weight_vector(self):
		return self._global_weight_vector

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

	# set the consensused blockchain
	def set_blockchain(self, blockchain):
		self._blockchain = blockchain

	def is_miner(self):
		return self.is_miner


	''' Functions for Workers '''

	def worker_set_sample_size(self, sample_size):
		# technically miner does not need sample_size, but our program allows role change for every epoch, and sample_size will not change if change role back and forth. Thus, we will first set sample_size for the device, no matter it is workder or miner, since it doesn't matter for miner to have this value as well. Same goes for step_size.
		# if self._is_miner:
		# 	print("Sample size is not required for miners.")
		# else:
		self._sample_size = sample_size

	def worker_set_step_size(self, step_size):
		# if self._is_miner:
		# 	print("Step size is only for workers to calculate weight updates.")
		# else:
		if step_size <= 0:
			print("Step size has to be positive.")
		else:
			self._step_size = step_size

	def worker_generate_dummy_data(self):
		# https://stackoverflow.com/questions/15451958/simple-way-to-create-matrix-of-random-numbers
		# if self._is_miner:
		# 	print("Warning - 
		# Device is initialized as miner.")
		# else:
		if not self._data:
			for _ in range(self._sample_size):
				self._data.append({'x': torch.randint(0, high=20, size=(self._data_dim, 1)), 'y': torch.randint(0, high=20, size=(1, 1))})
		else:
			print("The data of this worker has already been initialized. Changing data is not currently implemented in this version.")

	# worker global weight initialization or update
	def worker_set_global_weihgt(self, weight=None):
		if self._is_miner:
			print("Miner does not set weight values")
		else:
			if not self._global_weight_vector:
				# if not updating, initialize with all 0s, as directed by Dr. Park
				# Or, we should hard code a vector with some small values for the device class as it has to be the same for every device at the beginning
				self._global_weight_vector = torch.zeros(self._data_dim, 1)
			else:
				self._global_weight_vector = weight
	
	def worker_associate_minder(self):
		if self._is_miner:
			print("Miner does not associate with another miner.")
			return None
		else:
			global peers
			miner_nodes = []
			for node in peers:
				response = requests.get(f'{node}/get_role')
				if response.text == 'True':
					miner_nodes.append(node)
		# associate a random miner
		if miner_nodes:
			return random.choice(miner_nodes)
		else:
			# no device in this epoch is assigned as a miner
			return None
	
	def worker_upload_to_miner(self, upload, miner_address):
		if self._is_miner:
			print("Worker does not accept other workers' updates directly")
		else:
			miner_upload_endpoint = f"{miner_address}/new_transaction"
			requests.post(miner_upload_endpoint,
                  json=upload,
                  headers={'Content-type': 'application/json'})

	# BlockFL step 1 - train with regression
	# return local computation time, and delta_fk(wl) as a list
	# global_gradient is calculated after updating the global_weights
	def worker_local_update(self):
		if self._is_miner:
			print("Miner does not perfrom gradient calculations.")
		else:
			# SVRG algo, BlockFL section II and reference[4] 3.2
			# gradient of loss function chosen - mean squared error
			# delta_fk(wl) for each sk
			global_gradients_per_data_point = []
			# initialize the local weights as the current global weights
			local_weight = self._global_weight_vector
			# calculate delta_f(wl)
			last_block = self._blockchain.get_last_block()
			if last_block is not None:
				transactions = last_block.get_transactions()
				''' transactions = [{'device_id': _idx # used for debugging, updated_weigts': w, 'updated_gradients': [f1wl, f2wl ... fnwl]} ... ] '''
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
			# function(1)
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
				# record this value to upload
				global_gradients_per_data_point.append(delta_fk_wl)

				# calculate local update
				local_weight = local_weight - (step_size/len(self._data)) * (delta_fk_wil - delta_fk_wl + delta_f_wl)

			# device_id is not required. Just for debugging purpose
			return {"device_id": self._idx, "local_weight_update": local_weight, "global_gradients_per_data_point": global_gradients_per_data_point, "computation_time": time.time() - start_time}

	# TODO
	# def worker_global_update(self):

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
			# also set its hash as well. _block_hash is the same as pow proof
			candidate_block.set_hash()
			return current_hash
		else:
			print('Worker does not perform PoW.')
	
	# TODO cross-verification, method to verify updates, and make use of check_pow_proof
	# def cross-verification()

	def miner_receive_worker_updates(self, transaction):
		if self._is_miner:
			self._unmined_transactions.append(transaction)
		else:
			print("Worker cannot receive other workers' updates.")

	# TODO return pow_proof?
	def mine_transactions(self):
		if self._is_miner:
			if self._unmined_transactions:	
				# get the last block and construct the candidate block
				last_block = self._blockchain.get_last_block()

				candidate_block = Block(idx=last_block.get_block_idx+1,
				transactions=self._unmined_transactions,
				timestamp=time.time(),
				previous_hash=last_block.get_previous_hash())

				# mine the candidate block by PoW, inside which the _block_hash is also set
				pow_proof = self.proof_of_work(candidate_block)
				#TODO broadcast the block
		else:
			print("Worker does not mine transactions.")

	''' Common Methods '''

	# including adding the genesis block
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
            # 2. check if the proof is valid(_block_hash is also verified).
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
		return pow_proof.startswith('0' * Blockchain.difficulty) and pow_proof == block_to_check.compute_hash()

	''' consensus algorithm for the longest chain '''
	
	@classmethod
	def check_chain_validity(cls, chain_to_check):
		for block in chain_to_check[1:]:
			if cls.check_pow_proof(block, block.get_block_hash()) and block.get_previous_hash == chain_to_check[chain_to_check.index(block) - 1].get_block_hash():
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
		chain_len = len(self.blockchain.chain)

		for node in peers:
			response = requests.get(f'{node}/chain')
			length = response.json()['length']
			chain = response.json()['chain']
			if length > chain_len and blockchain.check_chain_validity(chain):
				# Longer valid chain found!
				chain_len = length
				longest_chain = chain

		if longest_chain:
			self.blockchain.chain = longest_chain
			return True

		return False

''' App Starts Here '''

app = Flask(__name__)

# pre-defined and agreed fields
DATA_DIM = 10
SAMPLE_SIZE = 20
STEP_SIZE = 3
EPSILON = 0.02
# miner waits for 180s to fill its candidate block with updates from devices
MINER_WAITING_TIME = 180

PROMPT = ">>>"

# the address to other participating members of the network
peers = set()

# create a device with a 4 bytes (8 hex chars) id
# the device's copy of blockchain also initialized
device = Device(binascii.b2a_hex(os.urandom(4)).decode('utf-8'))
# set data dimension
device.set_data_dim(DATA_DIM)

@app.route('/get_role', methods=['GET'])
def return_role():
	return "True" if device.is_miner() else "False"

# endpoint to for worker to upload updates to the associated miner
@app.route('/new_transaction', methods=['POST'])
def new_transaction():
    update_data = request.get_json()
    required_fields = ["device_id", "local_weight_update", "global_gradients_per_data_point", "computation_time"]

    for field in required_fields:
        if not update_data.get(field):
            return "Invalid transaction(update) data", 404

    update_data["timestamp"] = time.time()
    device.miner_receive_worker_updates(update_data)

    return "Success", 201


# start the app
# assign tasks based on role
# @app.route('/')
def runApp():

	print(f"{PROMPT} Device is setting sample size {SAMPLE_SIZE}")
	device.worker_set_sample_size(SAMPLE_SIZE)
	print(f"{PROMPT} Step size set to {STEP_SIZE}")
	device.worker_set_step_size(STEP_SIZE)
	print(f"{PROMPT} Device is generating the dummy data.")
	device.worker_generate_dummy_data()

	# TODO change to < EPSILON
	while True:
		print(f"Starting epoch {len(device.get_blockchain().chain)+1}")
		# assign/change role of this device
		device.set_miner()
		if device.is_miner():
			print(f"{PROMPT} This is Miner with ID {device.get_idx()}")
			# verify uploads? How?

		else:
			print(f"{PROMPT} This is workder with ID {device.get_idx()}")
			# while registering, chain was synced, if any
			print(f"{PROMPT} Workder is performing Step1 - local update")
			upload = device.worker_local_update()
			# worker associating with miner
			miner_address = device.worker_associate_minder()
			if device.worker_associate_minder() is not None:
				print(f"{PROMPT} Workder {device.get_idx()} now assigned to miner with address {miner_address}.")
			# worker uploads data to miner
			device.worker_upload_to_miner(upload, miner_address)


# endpoint to return the node's copy of the chain.
# Our application will be using this endpoint to query the contents in the chain to display
@app.route('/chain', methods=['GET'])
def query_blockchain():
    chain_data = []
    for block in device.get_blockchain().chain:
        chain_data.append(block.__dict__)
    return json.dumps({"length": len(chain_data),
                       "chain": chain_data,
                       "peers": list(peers)})

# TODO helper function used in register_with_existing_node() only while registering node
def sync_chain_from_dump(chain_dump):
    # generated_blockchain.create_genesis_block()
    for block_data in chain_dump:
        # if idx == 0:
        #     continue  # skip genesis block
        block = Block(block_data["_idx"],
                      block_data["_transactions"],
                      block_data["_timestamp"],
                      block_data["_previous_hash"],
                      block_data["_nonce"])
        pow_proof = block_data['_block_hash']
        added = device.add_block(block, pow_proof)
        if not added:
            raise Exception("The chain dump is tampered!!")
			# break
    # return generated_blockchain



''' add node to the network '''

# endpoint to add new peers to the network.
# why it's using POST here?
@app.route('/register_node', methods=['POST'])
def register_new_peers():
    node_address = request.get_json()["registerer_node_address"]
    if not node_address:
        return "Invalid data", 400

    # Add the node to the peer list
    peers.add(node_address)

    # Return the consensus blockchain to the newly registered node so that the new node can sync
    chain_meta = query_blockchain()
	return {"chain_meta": chain_meta, "global_weight_vector": device.get_global_weight_vector}


@app.route('/register_with', methods=['POST'])
def register_with_existing_node():
    """
    Internally calls the `register_node` endpoint to register current node with the node specified in the
    request, and sync the blockchain as well as peer data.
    """
    register_with_node_address = request.get_json()["register_with_node_address"]
    if not register_with_node_address:
        return "Invalid data", 400

    data = {"registerer_node_address": request.host_url}
    headers = {'Content-Type': "application/json"}

    # Make a request to register with remote node and obtain information
    response = requests.post(register_with_node_address + "/register_node", data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        # global blockchain
        global peers
        # sync the chain
        chain_data_dump = json.loads(response.json()['chain_meta'])['chain']
        sync_chain_from_dump(chain_data_dump)
		# sync the global weight from this register_with_node
		# TODO that might be just a string!!!
		global_weight_to_sync = response.json()['global_weight_vector']
		# update peer list according to the register-with node
        peers.update(json.loads(response.json()['chain_meta'])['peers'])
        return "Registration successful", 200
    else:
        # if something goes wrong, pass it on to the API response
        # return response.content, response.status_code, "why 404"
        return "weird"




















	
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

