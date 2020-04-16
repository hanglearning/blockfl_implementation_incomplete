import sys
import random
import time
import torch
import os
import binascii

import json
from hashlib import sha256

DEBUG_MODE = True # press any key to continue

# reference - https://developer.ibm.com/technologies/blockchain/tutorials/develop-a-blockchain-application-from-scratch-in-python/
class Block:
    def __init__(self, idx, transactions=[], generation_time=None, previous_hash=None, nonce=0):
        self._idx = idx
        self._transactions = transactions
        self._generation_time = generation_time
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

    def get_chain_length(self):
        return len(self.chain)

    def get_last_block(self):
        if len(self.chain) > 0:
            return self.chain[-1]
        else:
            # blockchain doesn't even have its genesis block
            return None

class Worker:
	def __init__(self, idx):
		self._idx = idx
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

    ''' getters '''
	# get device id
	def get_idx(self):
		return self._idx

	# get device's copy of blockchain
	def get_blockchain(self):
		return self._blockchain

    def get_current_epoch(self):
        return self._blockchain.get_chain_length()+1

    def get_data(self):
        return self._data

	# get global_weight_vector, used while being the register_with node to sync with the registerer node
	def get_global_weight_vector(self):
		return self._global_weight_vector

    ''' setters '''
	# set data dimension
	def set_data_dim(self, data_dim)
		self._data_dim = data_dim

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
			miner_nodes = set()
			for node in peers:
				response = requests.get(f'{node}/get_role')
				if response.status_code == 200:
					if response.text == 'Miner':
						miner_nodes.add(node)
				else:
					return "Error in worker_associate_minder()", response.status_code
		# associate a random miner
		if miner_nodes:
			return random.sample(miner_nodes, 1)[0]
		else:
			# no device in this epoch is assigned as a miner
			return None
	
    # TODO
	def worker_upload_to_miner(self, upload, miner_address):
		if self._is_miner:
			print("Worker does not accept other workers' updates directly")
		else:
            checked = False
            # check if node is still a miner
            response = requests.get(f'{miner_address}/get_role')
				if response.status_code == 200:
					if response.text == 'Miner':
                        # check if worker and miner are in the same epoch
                        response_epoch = requests.get(f'{miner_address}/return_miner_epoch')
                        if response_epoch.status_code == 200:
                            miner_epoch = int(response_epoch.text)
                            if miner_epoch == self.get_current_epoch():
                                checked = True
                            else:
                                # TODO not performing the same epoch, resync the chain
                                # consensus()?
            if checked:
                # check if miner is within the wait time of accepting updates
                response_miner_accepting = requests.get(f'{miner_address}/within_miner_wait_time')
                if response_miner_accepting.status_code == 200:
                    if response_miner_accepting.text == "True":
                        miner_upload_endpoint = f"{miner_address}/new_transaction"
                        requests.post(miner_upload_endpoint,
                            json=upload,
                            headers={'Content-type': 'application/json'})
                    else:
                        # TODO What to do next?
                        return "Not within miner waiting time."
                else:
                    return "Error getting miner waiting status", response_miner_accepting.status_code
            
			

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
	def worker_global_update(self):
		transactions_in_downloaded_block = self._blockchain.get_last_block().get_transactions()
		Ni = SAMPLE_SIZE
		Ns = len(transactions_in_downloaded_block)*Ni
		global_weight_tensor_accumulator = torch.zeros_like(self._global_weight_vector)
		for update in transactions_in_downloaded_block:
			updated_weigts = update["updated_weigts"]
			tensor_accumulator += (Ni/Ns)*(updated_weigts - self._global_weight_vector)
		self._global_weight_vector += global_weight_tensor_accumulator
		print("Global Update Done.")

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

PROMPT = ">>>"

# the address to other participating members of the network
peers = set()

# create a worker with a 4 bytes (8 hex chars) id
# the device's copy of blockchain also initialized
device = Worker(binascii.b2a_hex(os.urandom(4)).decode('utf-8'))

@app.route('/get_role', methods=['GET'])
def return_role():
	return "Worker"

@app.route('/get_worker_data', methods=['GET'])
def return_data():
	json.dumps({"data": device.get_data()})

# used while miner check for block size in miner_receive_worker_updates()
@app.route('/get_worker_epoch', methods=['GET'])
def get_worker_epoch():
	if not device.is_miner:
		return str(device.get_current_epoch())
	else:
		# TODO make return more reasonable
		return "error"

# start the app
# assign tasks based on role
# @app.route('/')
def runApp():

    print(f"{PROMPT} Device is setting data dimensionality {DATA_DIM}")
    device.set_data_dim(DATA_DIM)
	print(f"{PROMPT} Device is setting sample size {SAMPLE_SIZE}")
	device.worker_set_sample_size(SAMPLE_SIZE)
	print(f"{PROMPT} Step size set to {STEP_SIZE}")
	device.worker_set_step_size(STEP_SIZE)
	print(f"{PROMPT} Device is generating the dummy data.")
	device.worker_generate_dummy_data()

	# TODO change to < EPSILON
	while True:
        print(f"{PROMPT} This is workder with ID {device.get_idx()}")
        print(f"Starting epoch {device.get_current_epoch()}...")
        # while registering, chain was synced, if any
		if DEBUG_MODE:
            cont = input("Next worker_local_update. Continue?")
        print(f"{PROMPT} Worker is performing Step1 - local update...")
        upload = device.worker_local_update()
        # used for debugging
        print("Local updates done.")
        print(f"local_weight_update: {upload["local_weight_update"]}")
        print(f"global_gradients_per_data_point: {upload["global_gradients_per_data_point"]}")
        print(f"computation_time: {upload["computation_time"]}")
        # worker associating with miner
		if DEBUG_MODE:
            cont = input("Next worker_associate_minder. Continue?")
        miner_address = device.worker_associate_minder()
        while device.worker_associate_minder() is not None:
            print(f"{PROMPT} This workder {device.get_idx()} now assigned to miner with address {miner_address}.")
            # worker uploads data to miner
            device.worker_upload_to_miner(upload, miner_address)
        else:
            wait_new_miner_time = 10
            print(f"No miner in peers yet. Re-requesting miner address in {wait_new_miner_time} secs")
            time.sleep(wait_new_miner_time)
            miner_address = device.worker_associate_minder()
		# TODO during this time period the miner may request the worker to download the block and finish global updating. Need thread programming!
		if DEBUG_MODE:
            cont = input("Next sleep 180. Continue?")
		time.sleep(180)
        
@app.route('/download_block_from_miner', methods=['POST'])
def download_block_from_miner():
    downloaded_block = request.get_json()["block_to_download"]
	pow_proof = request.get_json()["pow_proof"]
	# TODO may need to construct block from dump!
	device.add_block(downloaded_block, pow_proof)
	# TODO proper way to trigger global update??
	device.worker_global_update()
    return "Success", 201

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
                      block_data["_generation_time"],
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
