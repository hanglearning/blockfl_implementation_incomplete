from flask import Flask, request
import requests
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
        
    # remove time_stamp
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
    
    # setters
    def set_previous_hash(self, hash_to_set):
        self._previous_hash = hash_to_set

    def add_verified_transaction(self, transaction):
        # after verified in cross_verification()
        self._transactions.append(transaction)

    def set_generation_time(self):
        self._generation_time = time.time()

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
        self._received_transactions = []
        # used to broadcast block for workers to do global updates
        self._associated_workers = set()
        # used to check if block size is full
        self._current_epoch_worker_nodes = set()
        # used in miner_broadcast_updates and block propogation
        self._current_epoch_miner_nodes = set()
        self._received_updates_from_miners = []
        # used in cross_verification and in the future PoS
        self._rewards = 0

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

    def get_rewards(self, rewards):
        self._rewards += rewards

    def associated_worker(self, worker_address):
        self._associated_workers.add(worker_address)

    def reset_associated_workers(self):
        self._associated_workers.clear()

    def get_all_current_epoch_workers(self):
        print("get_all_current_epoch_workers() called", self._current_epoch_worker_nodes)
        self._current_epoch_worker_nodes.clear()
        for node in peers:
            response = requests.get(f'{node}/get_role')
            if response.status_code == 200:
                if response.text == 'Worker':
                    response2 = requests.get(f'{node}/get_worker_epoch')
                    if response2.status_code == 200:
                        if int(response2.text) == self.get_current_epoch():
                            self._current_epoch_worker_nodes.add(node)
            else:
                return response.status_code
        if DEBUG_MODE:
            print("After get_all_current_epoch_workers() called", self._current_epoch_worker_nodes)
        
    def get_all_current_epoch_miners(self):
        self._current_epoch_miner_nodes.clear()
        for node in peers:
            response = requests.get(f'{node}/get_role')
            if response.status_code == 200:
                if response.text == 'Miner':
                    response2 = requests.get(f'{node}/get_miner_epoch')
                    if response2.status_code == 200:
                        if int(response2.text) == device.get_current_epoch():
                            self._current_epoch_miner_nodes.append(node)
            else:
                return response.status_code
        if DEBUG_MODE:
            print("get_all_current_epoch_miners() called", self._current_epoch_miner_nodes)

    def clear_received_updates_for_new_epoch(self):
        self._received_transactions.clear()
        if DEBUG_MODE:
            print("clear_received_updates_for_new_epoch() called", self._received_transactions)

    def clear_received_updates_from_other_miners(self):
        self._received_updates_from_miners.clear()
        if DEBUG_MODE:
            print("clear_received_updates_from_other_miners() called", self._received_updates_from_miners)

    def add_received_updates_from_miners(self, one_miner_updates):
        self._received_updates_from_miners.append(one_miner_updates)

    def request_associated_workers_download(self, pow_proof):
        block_to_download = self._blockchain.get_last_block()
        data = {"block_to_download": block_to_download, "pow_proof": pow_proof}
        headers = {'Content-Type': "application/json"}
        for worker in self._associated_workers:
            response = requests.post(f'{worker}/download_block_from_miner', data=json.dumps(data), headers=headers)
            if response.status_code == 200:
                print(f'Requested Worker {worker} to download the block.')
            
    
    # TODO rewards
    # TODO need a timer
    def cross_verification(self):
        if DEBUG_MODE:
            print("cross_verification() called")
        # Block index starting at 0
        candidate_block = Block(idx=self._blockchain.get_chain_length())
        # diff miners most likely have diff generation time but we add the one that's mined
        candidate_block.set_generation_time()        
        # it makes sense to first verify the updates itself received
        # verification machenism not specified in paper, so here we only verify the data_dim
        if self._received_transactions:
            for update in self._received_transactions:
                if len(update['local_weight_update']) == DATA_DIM:
                    candidate_block.add_verified_transaction(update)
                else:
                    pass
                self.get_rewards(1)
        # cross-verification
        if self._received_updates_from_miners:
            for update in self._received_updates_from_miners:
                if len(update['received_updates']['local_weight_update']) == DATA_DIM:
                    candidate_block.add_verified_transaction(update)
                else:
                    pass
                self.get_rewards(1)

        if DEBUG_MODE:
            print("Rewards after cross_verification", self._rewards)
            print("candidate_block", candidate_block)

        # when timer ends
        return candidate_block

    # TODO or rewards here?
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
            return current_hash, candidate_block
        else:
            print('Worker does not perform PoW.')

    def miner_receive_worker_updates(self, transaction):
        if self._is_miner:
            self._received_transactions.append(transaction)
            print(f"Miner {self.get_idx} received updates from {transaction['device_id']}")
            # check block size
            if len(self._current_epoch_worker_nodes) == len(self._received_transactions):
                # TODO abort the timer in miner_set_wait_time()
                # https://stackoverflow.com/questions/5114292/break-interrupt-a-time-sleep-in-python
                pass
        else:
            print("Worker cannot receive other workers' updates.")

    def miner_broadcast_updates(self):
        # get all miners in this epoch
        self.get_all_current_epoch_miners()

        data = {"miner_id": self._idx, "received_updates": self._received_transactions}
        headers = {'Content-Type': "application/json"}

        # broadcast the updates
        for miner in self._current_epoch_miner_nodes:
            response = requests.post(miner + "/receive_updates_from_miner", data=json.dumps(data), headers=headers)
            if response.status_code == 200:
                print(f'Miner {self._idx} sent unconfirmed updates to miner {miner}')

    def miner_propogate_the_block(self, block_to_propogate, pow_proof):
        # refresh the miner nodes in case there's any gone offline
        device.get_all_current_epoch_miners()

        data = {"miner_id": self._idx, "propogated_block": block_to_propogate, "pow_proof": pow_proof}
        headers = {'Content-Type': "application/json"}

        for miner in self._current_epoch_miner_nodes:
            # Make a request to register with remote node and obtain information
            response = requests.post(miner + "/receive_propogated_block", data=json.dumps(data), headers=headers)
            if response.status_code == 200:
                print(f'Miner {self._idx} sent the propogated block to miner {miner}')

    # TODO THIS FUNCTION MUST ABORT IF RECEIVED A BLOCK FROM ANOTHER MINER!!!
    def miner_mine_block(self, block_to_mine):
        if self._is_miner:
            if block_to_mine.get_transactions():
                # TODO
                # get the last block and add previous hash
                last_block = self._blockchain.get_last_block()

                block_to_mine.set_previous_hash(last_block.get_block_hash())
                # TODO
                # mine the candidate block by PoW, inside which the _block_hash is also set
                pow_proof, mined_block = self.proof_of_work(block_to_mine)
                # propagate the block in main()

                if DEBUG_MODE:
                    print("miner_mine_block() called", pow_proof, mined_block)
                return pow_proof, mined_block
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
DATA_DIM = 4
SAMPLE_SIZE = 3
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

        update_data["generation_time"] = time.time()
        device.miner_receive_worker_updates(update_data)
        if DEBUG_MODE:
            print("new_transaction() called from a worker", update_data)
    return "Success", 201

@app.route('/receive_updates_from_miner', methods=['POST'])
def receive_updates_from_miner():
    miner_id = request.get_json()["miner_id"]
    with_updates = request.get_json()["received_updates"]
    one_miner_updates = {"from_miner": miner_id, "received_updates": with_updates}
    device.add_received_updates_from_miners(one_miner_updates)
    print(f"Received updates from miner {miner_id}.")
    
@app.route('/receive_propogated_block', methods=['POST'])
def receive_propogated_block():
    # TODO ABORT THE RUNNING POW!!!
    miner_id = request.get_json()["miner_id"]
    pow_proof = request.get_json()["pow_proof"]
    propogated_block = request.get_json()["propogated_block"]
    # TODO may have to construct the block from dump here!!!
    # # check pow proof
    # if pow_proof.startswith('0' * Blockchain.difficulty) and pow_proof == propogated_block.compute_hash(): DONE IN add_block
    # add this block to the chain
    device.add_block(propogated_block, pow_proof)


# start the app
# assign tasks based on role
@app.route('/')
def runApp():
    # wait for worker maximum wating time
    while True:
        print(f"Starting epoch {device.get_current_epoch()}...")
        print(f"{PROMPT} This is Miner with ID {device.get_idx()}")
        if DEBUG_MODE:
            cont = input("Next get all workers in this epoch. Continue?")
        # get all workers in this epoch, used in miner_receive_worker_updates()
        device.get_all_current_epoch_workers()
        if DEBUG_MODE:
            cont = input("Next clear all received updates from the last epoch if any. Continue?")
        # clear all received updates from the last epoch if any
        device.clear_received_updates_for_new_epoch()
        device.clear_received_updates_from_other_miners()
        if DEBUG_MODE:
            cont = input("Next miner_set_wait_time(). Continue?")
        # waiting for worker's updates. While miner_set_wait_time() is working, miner_receive_worker_updates will check block size by checking and when #(tx) = #(workers), abort the timer 
        device.miner_set_wait_time()
        # miner broadcast received local updates
        if DEBUG_MODE:
            cont = input("Next miner_broadcast_updates(). Continue?")
        device.miner_broadcast_updates()
        # TODO find a better approach to implement, maybe use thread - wait for 180s to receive updates from other miners. Also need to consider about the block size!!
        if DEBUG_MODE:
            cont = input("Next time.sleep(180). Continue?")
        time.sleep(180)
        # start cross-verification
        # TODO verify uploads? How?
        if DEBUG_MODE:
            cont = input("Next cross_verification. Continue?")
        candidate_block = device.cross_verification()
        # miner mine transactions by PoW on this candidate_block
        if DEBUG_MODE:
            cont = input("Next miner_mine_block. Continue?")
        pow_proof, mined_block = device.miner_mine_block(candidate_block)
        # block_propagation
        # TODO if miner_mine_block returns none, which means it gets aborted, then it does not run propogate_the_block and add its own block. If not, run the next two.
        if DEBUG_MODE:
            cont = input("Next miner_propogate_the_block. Continue?")
        device.miner_propogate_the_block(mined_block, pow_proof)
        # add its own block
        # TODO fork ACK?
        if DEBUG_MODE:
            cont = input("Next add_block. Continue?")
        device.add_block(candidate_block)
        # send updates to its associated miners
        if DEBUG_MODE:
            cont = input("Next request_associated_workers_download. Continue?")
        device.request_associated_workers_download(pow_proof)


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
    if DEBUG_MODE:
            print("register_new_peers() called, peers", repr(peers))
    # Return the consensus blockchain to the newly registered node so that the new node can sync
    return {"chain_meta": query_blockchain()}


@app.route('/register_with', methods=['POST'])
def register_with_existing_node():
    """
    Internally calls the `register_node` endpoint to register current node with the node specified in the
    request, and sync the blockchain as well as peer data.
    """
    register_with_node_address = request.get_json()["register_with_node_address"]
    if not register_with_node_address:
        return "Invalid request - must specify a register_with_node_address!", 400

    data = {"registerer_node_address": request.host_url}
    headers = {'Content-Type': "application/json"}

    # Make a request to register with remote node and obtain information
    response = requests.post(register_with_node_address + "/register_node", data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        # global blockchain
        global peers
        # add the register_with_node_address as a peer
        peers.add(register_with_node_address)
        # sync the chain
        chain_data_dump = json.loads(response.json()['chain_meta'])['chain']
        sync_chain_from_dump(chain_data_dump)
        
        # NO NO NO sync the global weight from this register_with_node
        # TODO that might be just a string!!!
        # global_weight_to_sync = response.json()['global_weight_vector']
        # change to let node calculate global_weight_vector block by block

        # update peer list according to the register-with node
        peers.update(json.loads(response.json()['chain_meta'])['peers'])
        # remove itself if there is
        peers.remove(request.host_url)
        if DEBUG_MODE:
            print("register_with_existing_node() called, peers", repr(peers))
        return "Registration successful", 200
    else:
        # if something goes wrong, pass it on to the API response
        # return response.content, response.status_code, "why 404"
        return "weird"

# TODO
# block add time can use another list to store if necessary


''' debug methods '''
# debug peer var
@app.route('/debug_peers', methods=['GET'])
def debug_peers():
    return repr(peers)