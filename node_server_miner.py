import pdb

from flask import Flask, request
import requests
import sys
import random
import time
import torch
import os
import binascii
import copy

import json
from hashlib import sha256
DEBUG_MODE = True # press any key to continue


# reference - https://developer.ibm.com/technologies/blockchain/tutorials/develop-a-blockchain-application-from-scratch-in-python/
class Block:
    # https://stackoverflow.com/questions/3161827/what-am-i-doing-wrong-python-object-instantiation-keeping-data-from-previous-in
    # transactions=[] causing errors. Do not pass mutable object as a default value
    # def __init__(self, idx, transactions=[], block_generation_time=None, previous_hash=None, nonce=0):
    def __init__(self, idx, transactions=None, block_generation_time=None, previous_hash=None, nonce=0, block_hash=None):
        self._idx = idx
        self._transactions = transactions or []
        self._block_generation_time = block_generation_time
        self._previous_hash = previous_hash
        self._nonce = nonce
        # the hash of the current block, calculated by compute_hash
        self._block_hash = block_hash
        
    # remove time_stamp?
    def compute_hash(self, hash_previous_block=False):
        if hash_previous_block:
            block_content = self.__dict__
        else:
            block_content = copy.deepcopy(self.__dict__)
            block_content['_block_hash'] = None
        block_content = json.dumps(block_content, sort_keys=True)
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
    
    # def remove_block_hash_to_verify_pow(self):
    #     self._block_hash = None
    
    # setters
    def set_previous_hash(self, hash_to_set):
        self._previous_hash = hash_to_set

    def add_verified_transaction(self, transaction):
        # after verified in cross_verification()
        self._transactions.append(transaction)

    def set_block_generation_time(self):
        self._block_generation_time = time.time()

class Blockchain:

    # for PoW
    difficulty = 2

    def __init__(self):
        # it is fine to use a python list to store the chain for now
        # technically this should be _chain as well
        self._chain = []

    def get_chain_length(self):
        return len(self._chain)

    def get_last_block(self):
        if len(self._chain) > 0:
            return self._chain[-1]
        else:
            # blockchain doesn't even have its genesis block
            return None

    def append_block(self, block):
        self._chain.append(block)

class Miner:
    def __init__(self, idx):
        self._idx = idx
        self._is_miner = True
        # miner can also maintain the chain, tho the paper does not mention, but we think miner can be tranferred to worked any time back and forth, and also miner can use this info to obtain the epoch number to check if the uploaded updates from the worker are meant to be put into the same epoch
        self._blockchain = Blockchain()
        self._ip_and_port = None
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
        self._has_added_propogated_block = False
        self._propogated_block_pow = None

    ''' getters '''
    # get device id
    def get_idx(self):
        return self._idx

    # get device's copy of blockchain
    def get_blockchain(self):
        return self._blockchain

    def get_current_epoch(self):
        return self._blockchain.get_chain_length()+1

    def get_ip_and_port(self):
        return self._ip_and_port

    def is_propogated_block_added(self):
        return self._has_added_propogated_block

    def is_miner(self):
        return self.is_miner

    def get_propogated_block_pow(self):
        return self._propogated_block_pow

    ''' setters '''
    # set the consensused blockchain
    def set_blockchain(self, blockchain):
        self._blockchain = blockchain

    def set_ip_and_port(self, ip_and_port):
        self._ip_and_port = ip_and_port
    
    def propogated_block_has_been_added(self):
        self._has_added_propogated_block = True
    
    def set_propogated_block_pow(self, pow_proof):
        self._propogated_block_pow = pow_proof

    ''' Functions for Miners '''

    def get_rewards(self, rewards):
        self._rewards += rewards

    def associate_worker(self, worker_address):
        self._associated_workers.add(worker_address)

    def get_all_current_epoch_workers(self):
        print("get_all_current_epoch_workers() called", self._current_epoch_worker_nodes)
        # self._current_epoch_worker_nodes.clear()
        potential_new_peers = set()
        for node in peers:
            response = requests.get(f'{node}/get_role')
            if response.status_code == 200:
                if response.text == 'Worker':
                    response2 = requests.get(f'{node}/get_worker_epoch')
                    if response2.status_code == 200:
                        if int(response2.text) == self.get_current_epoch():
                            self._current_epoch_worker_nodes.add(node)
                            # side action - update (miner) peers from all workers
                            response3 = requests.get(f'{node}/get_peers')
                            if response3.status_code == 200:
                                potential_new_peers.update(response3.json()['peers'])
                            
            else:
                return response.status_code
        peers.update(potential_new_peers)
        try:
            peers.remove(self._ip_and_port)
        except:
            pass
        if DEBUG_MODE:
            print("After get_all_current_epoch_workers() called", self._current_epoch_worker_nodes)
        
    def get_all_current_epoch_miners(self):
        # self._current_epoch_miner_nodes.clear()
        potential_new_peers = set()
        for node in peers:
            response = requests.get(f'{node}/get_role')
            if response.status_code == 200:
                if response.text == 'Miner':
                    response2 = requests.get(f'{node}/get_miner_epoch')
                    if response2.status_code == 200:
                        if int(response2.text) == device.get_current_epoch():
                            self._current_epoch_miner_nodes.add(node)
                            # a chance to update peer list as well
                            response3 = requests.get(f'{node}/get_peers')
                            if response3.status_code == 200:
                                potential_new_peers.update(response3.json()['peers'])
            else:
                return response.status_code
        peers.update(potential_new_peers)
        try:
            peers.remove(self._ip_and_port)
        except:
            pass
        if DEBUG_MODE:
            print("get_all_current_epoch_miners() called", self._current_epoch_miner_nodes)

    # TODO should record epoch number in case accidentally remove updates from this epoch
    def clear_all_vars_for_new_epoch(self):
        # clear updates from workers and miners from the last epoch
        self._associated_workers.clear()
        self._current_epoch_worker_nodes.clear()
        self._current_epoch_miner_nodes.clear()
        self._received_transactions.clear()
        self._received_updates_from_miners.clear()
        self._has_added_propogated_block = False
        self._propogated_block_pow = None
        if DEBUG_MODE:
            print("clear_all_vars_for_new_epoch() called")


    def add_received_updates_from_miners(self, one_miner_updates):
        self._received_updates_from_miners.append(one_miner_updates)

    def request_associated_workers_download(self, pow_proof):
        block_to_download = self._blockchain.get_last_block()
        # pdb.set_trace()
        data = {"block_to_download": block_to_download.__dict__, "pow_proof": pow_proof}
        headers = {'Content-Type': "application/json"}
        if self._associated_workers:
            for worker in self._associated_workers:
                response = requests.post(f'{worker}/download_block_from_miner', data=json.dumps(data), headers=headers)
                if response.status_code == 200:
                    print(f'Requested Worker {worker} to download the block.')
        else:
            print("No associated workers this round. Begin Next epoch.")
            
    
    # TODO rewards
    # TODO need a timer
    def cross_verification(self):
        if DEBUG_MODE:
            print("cross_verification() called, initial rewards ", self._rewards)
            # pdb.set_trace()
        # Block index starting at 0
        candidate_block = Block(idx=self._blockchain.get_chain_length())
        # diff miners most likely have diff generation time but we add the one that's mined
        candidate_block.set_block_generation_time()        
        # it makes sense to first verify the updates itself received
        # verification machenism not specified in paper, so here we only verify the data_dim
        # pdb.set_trace()
        # if DEBUG_MODE:
        #     print("All block IDs")
        #     for block in self.get_blockchain()._chain:
        #         print(id(block))
        #     print("id of candidate_block", id(candidate_block))
        #     print("self._received_transactions", self._received_transactions)
        if self._received_transactions:
            for update in self._received_transactions:
                if len(update['local_weight_update']['update_tensor_to_list']) == DATA_DIM:
                    candidate_block.add_verified_transaction(update)
                else:
                    print("Error cross-verification SELF")
                self.get_rewards(DATA_DIM)
        print("After self-verification, rewards ", self._rewards)
        # cross-verification
        if self._received_updates_from_miners:
            for update_from_other_miner in self._received_updates_from_miners:
                # pdb.set_trace()
                for update in update_from_other_miner['received_updates']:
                    if len(update['local_weight_update']['update_tensor_to_list']) == DATA_DIM:
                        candidate_block.add_verified_transaction(update)
                    else:
                        print("Error cross-verification OTHER MINER")
                    self.get_rewards(DATA_DIM)
        if DEBUG_MODE:
            print("Rewards after cross_verification ", self._rewards)
            print("candidate_block", candidate_block)

        # when timer ends
        return candidate_block

    # TODO or rewards here?
    def proof_of_work(self, candidate_block):
        ''' Brute Force the nonce. May change to PoS by Dr. Jihong Park '''
        if DEBUG_MODE:
            print(f"Before PoW, block nonce is {candidate_block._nonce} and block hash is {candidate_block._block_hash}.")
        if self._is_miner:
            # pdb.set_trace()
            current_hash = candidate_block.compute_hash()
            while not current_hash.startswith('0' * Blockchain.difficulty):
                candidate_block.nonce_increment()
                current_hash = candidate_block.compute_hash()
            # return the qualified hash as a PoW proof, to be verified by other devices before adding the block
            # also set its hash as well. _block_hash is the same as pow proof
            candidate_block.set_hash()
            if DEBUG_MODE:
                print(f"After PoW, block nonce is {candidate_block._nonce} and block hash is {candidate_block._block_hash}.")
            return current_hash, candidate_block
        else:
            print('Worker does not perform PoW.')

    def miner_receive_worker_updates(self, transaction):
        if self._is_miner:
            # if DEBUG_MODE:
            #     print("self._received_transactions", self._received_transactions)
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
        # pdb.set_trace()
        # get all miners in this epoch
        self.get_all_current_epoch_miners()

        data = {"miner_id": self._idx, "received_updates": self._received_transactions}
        headers = {'Content-Type': "application/json"}

        # broadcast the updates
        for miner in self._current_epoch_miner_nodes:
            response = requests.post(miner + "/receive_updates_from_miner", data=json.dumps(data), headers=headers)
            if response.status_code == 200:
                print(f'Miner {self._idx} sent unconfirmed updates to miner {miner}')
        return "ok"

    def miner_propogate_the_block(self, block_to_propogate, pow_proof):
        # refresh the miner nodes in case there's any gone offline
        device.get_all_current_epoch_miners()

        data = {"miner_id": self._idx, "propogated_block": block_to_propogate.__dict__, "pow_proof": pow_proof}
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
                if last_block is None:
                    # mine the genesis block
                    block_to_mine.set_previous_hash(None)
                else:
                    block_to_mine.set_previous_hash(last_block.compute_hash(hash_previous_block=True))
                # TODO
                # mine the candidate block by PoW, inside which the _block_hash is also set
                pow_proof, mined_block = self.proof_of_work(block_to_mine)
                # propagate the block in main()
                # pdb.set_trace()
                if DEBUG_MODE:
                    print("miner_mine_block() called", pow_proof, mined_block)
                return pow_proof, mined_block
            else:
                print("No transaction to mine.")
                #TODO Skip or wait and go to the next epoch
        else:
            print("Worker does not mine transactions.")

    ''' Common Methods '''

    # including adding the genesis block
    def add_block(self, block_to_add, pow_proof):
        """
        A function that adds the block to the chain after two verifications(sanity check).
        """
        last_block = self._blockchain.get_last_block()
        # pdb.set_trace()
        if last_block is not None:
            # 1. check if the previous_hash referred in the block and the hash of latest block in the chain match.
            last_block_hash = last_block.compute_hash(hash_previous_block=True)
            if block_to_add.get_previous_hash() != last_block_hash:
                # to be used as condition check later
                return False
            # 2. check if the proof is valid(_block_hash is also verified).
            # remove its block hash to verify pow_proof as block hash was set after pow
            if not self.check_pow_proof(block_to_add, pow_proof):
                return False
            # All verifications done.
            self._blockchain.append_block(block_to_add)
            return True
        else:
            # only check 2. above
            if not self.check_pow_proof(block_to_add, pow_proof):
                return False
            # add genesis block
            self._blockchain.append_block(block_to_add)
            return True
    
    @staticmethod
    def check_pow_proof(block_to_check, pow_proof):
        # if not (block_to_add._block_hash.startswith('0' * Blockchain.difficulty) and block_to_add._block_hash == pow_proof): WRONG
        # shouldn't check the block_hash directly as it's not trustworthy and it's also private
        # pdb.set_trace()
        # Why this is None?
        # block_to_check_without_hash = copy.deepcopy(block_to_check).remove_block_hash_to_verify_pow()
        # block_to_check_without_hash = copy.deepcopy(block_to_check)
        # block_to_check_without_hash.remove_block_hash_to_verify_pow()

        return pow_proof.startswith('0' * Blockchain.difficulty) and pow_proof == block_to_check.compute_hash()

    ''' consensus algorithm for the longest chain '''
    
    # TODO Debug and write
    @classmethod
    def check_chain_validity(cls, chain_to_check):
        chain_len = chain_to_check.get_chain_length()
        if chain_len == 0:
            pass
        elif chain_len == 1:
            pass
        else:
            for block in chain_to_check[1:]:
                if cls.check_pow_proof(block, block.get_block_hash()) and block.get_previous_hash == chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_previous_block=True):
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
        chain_len = self._blockchain.get_chain_length()

        for node in peers:
            response = requests.get(f'{node}/get_chain_meta')
            length = response.json()['length']
            chain = response.json()['chain']
            if length > chain_len and self.check_chain_validity(chain):
                # Longer valid chain found!
                chain_len = length
                longest_chain = chain

        if longest_chain:
            self._blockchain._chain = longest_chain
            return True

        return False

''' App Starts Here '''

app = Flask(__name__)

# pre-defined and agreed fields
# miner use these values to verify data validity
DATA_DIM = 4
SAMPLE_SIZE = 3
# miner waits for 180s to fill its candidate block with updates from devices
MINER_WAITING_UPLOADS_PERIOD = 10

PROMPT = ">>>"

# TODO change to False and uncomment miner wait time
miner_accept_updates = True

# the address to other participating members of the network
peers = set()

# create a device with a 4 bytes (8 hex chars) id
# the device's copy of blockchain also initialized
device = Miner(binascii.b2a_hex(os.urandom(4)).decode('utf-8'))

def miner_set_wait_time():
    if device.is_miner():
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
    if device.is_miner():
        return str(device.get_current_epoch())
    else:
        # TODO make return more reasonable
        return "error"


# end point for worker to check whether the miner is now accepting the dates(within miner's wait time)
@app.route('/within_miner_wait_time', methods=['GET'])
def within_miner_wait_time():
    return "True" if miner_accept_updates else "False"

# endpoint to for worker to upload updates to the associated miner
# also miner will remember this worker in self._associated_workers
@app.route('/new_transaction', methods=['POST'])
def new_transaction():
    if miner_accept_updates:
        update_data = request.get_json()
        required_fields = ["device_id", "local_weight_update", "global_gradients_per_data_point", "computation_time", "this_worker_address"]

        for field in required_fields:
            if not update_data.get(field):
                return "Invalid transaction(update) data", 404
        device.associate_worker(update_data['this_worker_address'])
        peers.add(update_data['this_worker_address'])
        update_data["tx_received_time"] = time.time()
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
    return f"Received updates from miner {miner_id}."
    
@app.route('/receive_propogated_block', methods=['POST'])
def receive_propogated_block():
    # TODO ABORT THE RUNNING POW!!!
    if device.is_propogated_block_added():
        # TODO in paper it has fork is generated when miner receives a new block, but I think this should be generated when a received propogated block has been added to the chain
        return "A fork has happened."
    miner_id = request.get_json()["miner_id"]
    pow_proof = request.get_json()["pow_proof"]
    propogated_block = request.get_json()["propogated_block"]
    # first verify this block id == chain length
    block_idx = propogated_block["_idx"]
    if int(block_idx) != device.get_blockchain().get_chain_length():
        return "The received propogated block is not sync with this miner's epoch."
    # # check pow proof
    # if pow_proof.startswith('0' * Blockchain.difficulty) and pow_proof == propogated_block.compute_hash(): DONE IN add_block
    # add this block to the chain
    reconstructed_block = Block(block_idx,
                propogated_block["_transactions"],
                propogated_block["_block_generation_time"],
                propogated_block["_previous_hash"],
                propogated_block["_nonce"],
                propogated_block['_block_hash'])
    print("reconstructed_block", reconstructed_block.__dict__, "pow", pow_proof)
    # TODO Still verify dimension or not since this miner may not fully trust the other miner? If not verify, what's the reason?
    if device.add_block(reconstructed_block, pow_proof):
        device.set_propogated_block_pow(pow_proof)
        device.propogated_block_has_been_added()
        print("A propogated block has been mined and added to the blockchain.")
        return "A propogated block has been mined and added to the blockchain."


# start the app
# assign tasks based on role
@app.route('/')
def runApp():

    # wait for worker maximum wating time
    while True:
        print(f"==================")
        print(f"|  BlockFL Demo  |")
        print(f"==================\n")
        
        print(f"{PROMPT} This is Miner with ID {device.get_idx()}\n")
        print(f"Starting epoch {device.get_current_epoch()}...\n")
        #TODO recheck peer validity and remove offline peers
        # hopfully network delay won't cause bug if like the propogated block is added at the moment we clear _has_added_propogated_block to True. Though low odds, still have to think about it
        # if not device.is_propogated_block_added():
        if DEBUG_MODE:
            cont = input("First clear all related variables for the new epoch, including all received updates from the last epoch if any and associated workers and miners in order to start a new epoch. Continue?\n")
        # clear 5 vars
        device.clear_all_vars_for_new_epoch()
        # else:
        #     print("A propogated block has been added. Jump to request worker download.")
        #     pass
        
        if not device.is_propogated_block_added():
            if DEBUG_MODE:
                cont = input("Next get all workers in this epoch. Continue?\n")
            # get all workers in this epoch, used in miner_receive_worker_updates()
            device.get_all_current_epoch_workers()
        else:
            print("A propogated block has been added. Jump to request worker download.")
            pass
        
        if not device.is_propogated_block_added():
            if DEBUG_MODE:
                cont = input("Next miner_set_wait_time() to wait for workers to upload. Continue?\n")
            # waiting for worker's updates. While miner_set_wait_time() is working, miner_receive_worker_updates will check block size by checking and when #(tx) = #(workers), abort the timer 
            # TODO uncomment in production miner_set_wait_time()
        else:
            print("A propogated block has been added. Jump to request worker download.")
            pass
        
        if not device.is_propogated_block_added():
            # miner broadcast received local updates
            if DEBUG_MODE:
                cont = input("Next miner_broadcast_updates(). Continue?\n")
            device.miner_broadcast_updates()
        else:
            print("A propogated block has been added. Jump to request worker download.")
            pass

        if not device.is_propogated_block_added():
            # TODO find a better approach to implement, maybe use thread - wait for 180s to receive updates from other miners. Also need to consider about the block size!!
            if DEBUG_MODE:
                cont = input("Next time.sleep(180) to receive the propogated updates. Continue?\n")
            # time.sleep(180)
            # start cross-verification
        else:
            print("A propogated block has been added. Jump to request worker download.")
            pass

        if not device.is_propogated_block_added():
            # TODO verify uploads? How?
            if DEBUG_MODE:
                cont = input("Next cross_verification. Continue?\n")
            candidate_block = device.cross_verification()
        else:
            print("A propogated block has been added. Jump to request worker download.")
            pass

        if not device.is_propogated_block_added():
            # miner mine transactions by PoW on this candidate_block
            if DEBUG_MODE:
                cont = input("Next miner_mine_block. Continue?\n")
            pow_proof, mined_block = device.miner_mine_block(candidate_block)
        else:
            print("A propogated block has been added. Jump to request worker download.")
            pass
        
        if not device.is_propogated_block_added():
            # block_propagation
            # TODO if miner_mine_block returns none, which means it gets aborted, then it does not run propogate_the_block and add its own block. If not, run the next two.
            if DEBUG_MODE:
                cont = input("Next miner_propogate_the_block. Continue\n?")
            device.miner_propogate_the_block(mined_block, pow_proof)
        else:
            print("A propogated block has been added. Jump to request worker download.")
            pass

        if not device.is_propogated_block_added():
            # add its own block
            # TODO fork ACK?
            if DEBUG_MODE:
                cont = input("Next add_block. Continue?\n")
            if device.add_block(mined_block, pow_proof):
                print("Its own block has been added.")
        else:
            print("A propogated block has been added. Jump to request worker download.")
            pass

        # send updates to its associated miners
        if DEBUG_MODE:
            cont = input("Next request_associated_workers_download. Continue?\n")
            if device.is_propogated_block_added():
                # download the added propogated block
                device.request_associated_workers_download(device.get_propogated_block_pow())
            else:
                # download its own block
                device.request_associated_workers_download(pow_proof)
                if DEBUG_MODE:
                    cont = input("Next epoch. Continue?\n")


# endpoint to return the node's copy of the chain.
# Our application will be using this endpoint to query the contents in the chain to display
@app.route('/chain', methods=['GET'])
def display_chain():
    chain = json.loads(query_blockchain())["chain"]
    for block_iter in range(len(chain)):
        print(f"Block #{block_iter+1}")
        block = chain[block_iter]
        print("_idx", block["_idx"])
        for tx_iter in range(len(block["_transactions"])):
            print(f"\nTransaction {tx_iter}\n", block["_transactions"][tx_iter], "\n")
        print("_block_generation_time", block["_block_generation_time"])
        print("_previous_hash", block["_previous_hash"])
        print("_nonce", block["_nonce"])
        print("_block_hash", block["_block_hash"])
    return "Chain Returned in Port Console"

@app.route('/get_chain_meta', methods=['GET'])
def query_blockchain():
    chain_data = []
    for block in device.get_blockchain()._chain:
        chain_data.append(block.__dict__)
    return json.dumps({"length": len(chain_data),
                       "chain": chain_data,
                       "peers": list(peers)})


@app.route('/get_peers', methods=['GET'])
def query_peers():
    return json.dumps({"peers": list(peers)})


# TODO helper function used in register_with_existing_node() only while registering node
def sync_chain_from_dump(chain_dump):
    print("sync_chain_from_dump() called by miner")
    # generated_blockchain.create_genesis_block()
    for block_data in chain_dump:
        # if idx == 0:
        #     continue  # skip genesis block
        block = Block(block_data["_idx"],
                      block_data["_transactions"],
                      block_data["_block_generation_time"],
                      block_data["_previous_hash"],
                      block_data["_nonce"])
        pow_proof = block_data['_block_hash']
        # in add_block, check if pow_proof and previous_hash fileds both are valid
        added = device.add_block(block, pow_proof)
        if not added:
            raise Exception("The chain dump is tampered!!")
        else:
            pass
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

    transferred_this_node_address = request.get_json()["registerer_with_node_address"]
    if device.get_ip_and_port() == None:
        # this is a dirty hack for the first node in the network to set its ip and node and used to remove itself from peers
        device.set_ip_and_port(transferred_this_node_address)
    if device.get_ip_and_port() != transferred_this_node_address:
        return "This should never happen"

    # Add the node to the peer list
    peers.add(node_address)
    if DEBUG_MODE:
            print("register_new_peers() called, peers", repr(peers))
    # Return the consensus blockchain to the newly registered node so that the new node can sync
    return query_blockchain()


@app.route('/register_with', methods=['POST'])
def register_with_existing_node():
    """
    Internally calls the `register_node` endpoint to register current node with the node specified in the
    request, and sync the blockchain as well as peer data.
    """
    # assign ip and port for itself, mainly used to remove itself from peers list
    device.set_ip_and_port(request.host_url[:-1])

    register_with_node_address = request.get_json()["register_with_node_address"]
    if not register_with_node_address:
        return "Invalid request - must specify a register_with_node_address!", 400
    data = {"registerer_node_address": request.host_url[:-1], "registerer_with_node_address": register_with_node_address}
    headers = {'Content-Type': "application/json"}

    # Make a request to register with remote node and obtain information
    response = requests.post(register_with_node_address + "/register_node", data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        # global blockchain
        global peers
        # add the register_with_node_address as a peer
        peers.add(register_with_node_address)
        # sync the chain
        chain_data_dump = response.json()['chain']
        sync_chain_from_dump(chain_data_dump)
        
        # NO NO NO sync the global weight from this register_with_node
        # TODO that might be just a string!!!
        # global_weight_to_sync = response.json()['global_weight_vector']
        # change to let node calculate global_weight_vector block by block

        # update peer list according to the register-with node
        peers.update(response.json()['peers'])
        # remove itself if there is
        try:
            if DEBUG_MODE:
                print("Self IP and Port", device.get_ip_and_port())
            peers.remove(device.get_ip_and_port())
        except:
            pass
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
