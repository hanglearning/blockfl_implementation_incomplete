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

    def get_chain_length(self):
        return len(self.chain)

    def get_last_block(self):
        if len(self.chain) > 0:
            return self.chain[-1]
        else:
            # blockchain doesn't even have its genesis block
            return None

class Miner:
	def __init__(self, idx):
		self._idx = idx
		self._is_miner = True
		# miner can also maintain the chain, tho the paper does not mention, but we think miner can be tranferred to worked any time back and forth, and also miner can use this info to obtain the epoch number to check if the uploaded updates from the worker are meant to be put into the same epoch
		self._blockchain = Blockchain()
		''' attributes for miners '''
		self._unmined_transactions = []

    ''' getters '''
	# get device id
	def get_idx(self):
		return self._idx

	# get device's copy of blockchain
	def get_blockchain(self):
		return self._blockchain

    def get_current_epoch(self):
        return self._blockchain.get_chain_length()+1

    ''' setters '''
	# set the consensused blockchain
	def set_blockchain(self, blockchain):
		self._blockchain = blockchain

	def is_miner(self):
		return self.is_miner

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
            print(f"Miner {self.get_idx} received updates from {transaction['device_id']}")
            # check block size
            if len(current_epoch_worker_nodes) == len(self._unmined_transactions):
                # TODO abort the timer in miner_set_wait_time()
                # https://stackoverflow.com/questions/5114292/break-interrupt-a-time-sleep-in-python
                pass
		else:
			print("Worker cannot receive other workers' updates.")

    def miner_broadcast_updates(self):
        # get all workers in this epoch, used in miner_receive_worker_updates()
        current_epoch_miner_nodes = set()
        # START FROM HERE 4/15
        # for node in peers:
        #     response = requests.get(f'{node}/get_role')
        #     if response.status_code == 200:
        #         if response.text == 'Worker':
        #             response2 = requests.get(f'{node}/get_worker_epoch')
        #             if response2.status_code == 200:
        #                 if int(response2.text) == device.get_current_epoch():
        #                     current_epoch_worker_nodes.append(node)
        #     else:
        #         return response.status_code

	# TODO return pow_proof?
	def miner_mine_transactions(self):
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
				print("No transaction to mine.")
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
# miner use these values to verify data validity
DATA_DIM = 10
SAMPLE_SIZE = 20
# miner waits for 180s to fill its candidate block with updates from devices
MINER_WAITING_UPLOADS_PERIOD = 180

PROMPT = ">>>"

miner_accept_updates = False

# the address to other participating members of the network
peers = set()

# create a device with a 4 bytes (8 hex chars) id
# the device's copy of blockchain also initialized
device = Miner(binascii.b2a_hex(os.urandom(4)).decode('utf-8'))

def miner_set_wait_time():
	if device.is_miner:
		global miner_accept_updates
		miner_accept_updates = True
		print(f"{PROMPT} Miner wait time set to {MINER_WAITING_UPLOADS_PERIOD}s, waiting for updates...")
		time.sleep(MINER_WAITING_UPLOADS_PERIOD)
		miner_accept_updates = False
        print(f"{PROMPT} Miner done accepting updates in this epoch.")
	else:
		# TODO make return more reasonable
		return "error"

@app.route('/get_role', methods=['GET'])
def return_role():
	return "Miner"


# used while worker uploading the updates. 
# If epoch doesn't match, one of the entity has to resync the chain
@app.route('/get_miner_epoch', methods=['GET'])
def get_miner_epoch():
	if device.is_miner:
		return str(device.get_current_epoch())
	else:
		# TODO make return more reasonable
		return "error"


# end point for worker to check whether the miner is now accepting the dates(within miner's wait time)
@app.route('/within_miner_wait_time', methods=['GET'])
def within_miner_wait_time():
	return "True" if miner_accept_updates else "False"

# endpoint to for worker to upload updates to the associated miner
@app.route('/new_transaction', methods=['POST'])
def new_transaction():
    if miner_accept_updates:
        update_data = request.get_json()
        required_fields = ["device_id", "local_weight_update", "global_gradients_per_data_point", "computation_time"]

        for field in required_fields:
            if not update_data.get(field):
                return "Invalid transaction(update) data", 404

        update_data["timestamp"] = time.time()
        device.miner_receive_worker_updates(update_data)

    return "Success", 201

current_epoch_worker_nodes = set()
# start the app
# assign tasks based on role
# @app.route('/')
def runApp():
    # get all workers in this epoch, used in miner_receive_worker_updates()
    global current_epoch_worker_nodes
    for node in peers:
        response = requests.get(f'{node}/get_role')
        if response.status_code == 200:
            if response.text == 'Worker':
                response2 = requests.get(f'{node}/get_worker_epoch')
                if response2.status_code == 200:
                    if int(response2.text) == device.get_current_epoch():
                        current_epoch_worker_nodes.add(node)
        else:
            return response.status_code

    # wait for worker maximum wating time
	while True:
		print(f"Starting epoch {len(device.get_blockchain.chain)+1}...")
		# assign/change role of this device
		
        print(f"{PROMPT} This is Miner with ID {device.get_idx()}")
        # waiting for worker's updates 
        # while miner_set_wait_time() is working, miner_receive_worker_updates will check block size by checking and when #(tx) = #(workers), abort the timer 
        device.miner_set_wait_time()
        # miner broadcast received local updates
        device.miner_broadcast_updates()
        # TODO verify uploads? How?

        # miner mine transactions


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