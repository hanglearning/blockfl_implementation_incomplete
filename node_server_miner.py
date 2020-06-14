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
from threading import Event
from utils import *


import json
from hashlib import sha256

# https://stackoverflow.com/questions/14888799/disable-console-messages-in-flask-server
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from block import Block
from blockchain import Blockchain

DEBUG_MODE = True # press any key to continue


class Miner(Device):
    def __init__(self, idx):
        Device.__init__(self, idx)
        ''' attributes for miners '''
        self._received_transactions = []
        # used to broadcast block for workers to do global updates
        # also used to check if block size is full
        self._associated_workers = set()
        # self._current_epoch_worker_nodes = set()
        # used in miner_broadcast_updates and block propogation
        self._current_epoch_miner_nodes = set()
        self._received_updates_from_miners = []
        # self._has_added_propagated_block = False
        self._propagated_block_pow = None

    ''' getters '''

    # def is_propagated_block_added(self):
    #     return self._has_added_propagated_block

    def is_miner(self):
        return self.is_miner

    def get_propagated_block_pow(self):
        return self._propagated_block_pow

    ''' setters '''

    # def propagated_block_has_been_added(self):
    #     self._has_added_propagated_block = True
    
    def set_propagated_block_pow(self, pow_proof):
        self._propagated_block_pow = pow_proof

    ''' Functions for Miners '''


    def associate_worker(self, worker_address):
        self._associated_workers.add(worker_address)


    # TODO should record epoch number in case accidentally remove updates from this epoch
    def reset_related_vars_for_new_epoch(self):
        # clear updates from workers and miners from the last epoch
        self._associated_workers.clear()
        #self._current_epoch_worker_nodes.clear()
        #self._current_epoch_miner_nodes.clear()
        self._received_transactions.clear()
        self._received_updates_from_miners.clear()
        # self._has_added_propagated_block = False
        self._propagated_block_pow = None
        self._jump_to_next_epoch = False
        # if DEBUG_MODE:
        #     print("reset_related_vars_for_new_epoch() called")


    def add_received_updates_from_miners(self, one_miner_updates):
        self._received_updates_from_miners.append(one_miner_updates)

    def request_associated_workers_download(self, pow_proof):
        block_to_download = self._blockchain.get_last_block()
        data = {"miner_id": self._idx, "miner_ip": self._ip_and_port, "block_to_download": block_to_download.__dict__, "pow_proof": pow_proof}
        headers = {'Content-Type': "application/json"}
        offline_nodes = set()
        if self._associated_workers:
            for worker in self._associated_workers:
                response = requests.post(f'{worker}/download_block_from_miner', data=json.dumps(data), headers=headers)
                node_online = False
                if response.status_code == 200:
                    node_online = True
                else:
                    node_online = self.retry_offline_peers(worker)
                    # WRONG! REPOST!
                if node_online:
                    print(f'Requested Worker {worker} to download the block.')
                    return
                else:
                    print(f'The associated worker {worker} goes offline.')
                    offline_nodes.add()
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
        # TODO make it more robust
        if self._received_transactions:
            print("\nVerifying received updates from associated workers...")
            for update in self._received_transactions:
                # if len(update['local_weight_update']['update_tensor_to_list']) == DATA_DIM and len(update['global_gradients_per_data_point']) == SAMPLE_SIZE:
                if len(update['local_weight_update']['update_tensor_to_list']) == DATA_DIM:
                    candidate_block.add_verified_transaction(update)
                    print(f"Updates from worker {update['worker_ip']}({update['worker_id']}) are verified.")
                    # print(f"This miner now sends rewards to the above worker for the provision of data by the SAMPLE_SIZE {SAMPLE_SIZE}")
                    # data = {"miner_id": self._idx, "miner_ip": self._ip_and_port, "rewards": SAMPLE_SIZE}
                    print(f"This miner now sends rewards to the above worker for the provision of data by the SAMPLE_SIZE {DATA_DIM}")
                    data = {"miner_id": self._idx, "miner_ip": self._ip_and_port, "rewards": DATA_DIM}
                    headers = {'Content-Type': "application/json"}
                    response = requests.post(f"{update['worker_ip']}/get_rewards_from_miner", data=json.dumps(data), headers=headers)
                    node_online = False
                    if response.status_code == 200:
                        node_online = True
                    else:
                        self.retry_offline_peers({update['worker_ip']})
                    if node_online:
                        print(f'Rewards sent!\n')
                    else:
                        # TODO - what if this worker goes offline? It needs a coin address, then private and public key is necessary. Rewards need to be recorded on chain.
                        pass
                else:
                    print("Error cross-verification SELF for this update. Update tossed.")
                    #TODO toss this updates
                self.get_rewards(DATA_DIM)
        else:
            print("\nNo associated workers or no updates received from the associated workers. Skipping self verification...")
        print("After/Skip self verification, total rewards ", self._rewards)
        # cross-verification
        if self._received_updates_from_miners:
            for update_from_other_miner in self._received_updates_from_miners:
                print("\nVerifying broadcasted updates from other miners...")
                # pdb.set_trace()
                for update in update_from_other_miner['received_updates']:
                    # if len(update['local_weight_update']['update_tensor_to_list']) == DATA_DIM and len(update['global_gradients_per_data_point']) == SAMPLE_SIZE:
                    if len(update['local_weight_update']['update_tensor_to_list']) == DATA_DIM:
                        candidate_block.add_verified_transaction(update)
                        print(f"Updates from miner {update_from_other_miner['from_miner_ip']}({update_from_other_miner['from_miner_id']}) for worker {update['worker_ip']}({update['worker_id']}) are verified.")
                    else:
                        print("Error cross-verification OTHER MINER. This update is tossed.")
                        #TODO toss this updates
                    self.get_rewards(DATA_DIM)
        else:
            print("\nNo broadcasted updates received from other miners. Skipping cross verification.")
        print("After/Skip cross verification, total rewards ", self._rewards)
        print("\nCross verifications done")
        # print("candidate_block", candidate_block)

        # when timer ends
        return candidate_block

    # TEST FOR CONVERGENCE
    # def cross_verification(self):
    #     print("SKIP REWARD FOR CONVERGENCE TEST")
    #     candidate_block = Block(idx=self._blockchain.get_chain_length())
    #     candidate_block.set_block_generation_time()
    #     if self._received_transactions:
    #         for update in self._received_transactions:
    #             candidate_block.add_verified_transaction(update)
    #     if self._received_updates_from_miners:
    #         for update_from_other_miner in self._received_updates_from_miners:
    #             for update in update_from_other_miner['received_updates']:
    #                 candidate_block.add_verified_transaction(update)
    #     return candidate_block

    # TODO or rewards here?
    def proof_of_work(self, candidate_block, starting_nonce=0):
        ''' Brute Force the nonce. May change to PoS by Dr. Jihong Park '''
        if DEBUG_MODE:
            print(f"Before PoW, block nonce is {candidate_block._nonce} and block hash is {candidate_block._block_hash}.")
        if self._is_miner:
            # pdb.set_trace()
            candidate_block.set_nonce(starting_nonce)
            current_hash = candidate_block.compute_hash()
            while not current_hash.startswith('0' * Blockchain.difficulty) and not has_received_propagated_block.is_set():
                candidate_block.nonce_increment()
                current_hash = candidate_block.compute_hash()
            # return the qualified hash as a PoW proof, to be verified by other devices before adding the block
            # also set its hash as well. _block_hash is the same as pow proof
            if has_received_propagated_block.is_set():
                print("Aborting PoW as a propagated block has been received.")
                return candidate_block.get_current_nonce()
            candidate_block.set_hash()
            if DEBUG_MODE:
                print(f"After PoW, block nonce is {candidate_block._nonce} and block hash is {candidate_block._block_hash}.")
                print("This block will only be added to this miner's chain if this miner has not\n 1. received a propagated block, and\n 2. successfully verified the PoW proof of the propagated block.\nThis block would be dumped if both above conditions are met.")
            return current_hash, candidate_block
        else:
            print('Worker does not perform PoW.')

    def miner_receive_worker_updates(self, transaction):
        if self._is_miner:
            # if DEBUG_MODE:
            #     print("self._received_transactions", self._received_transactions)
            self._received_transactions.append(transaction)
            print(f"\nThis miner {self.get_ip_and_port()}({self._idx}) received updates from worker {transaction['this_worker_address']}({transaction['worker_id']})")
            # check block size
            if len(self._associated_workers) == len(self._received_transactions):
                # TODO abort the timer in miner_set_wait_time()
                # https://stackoverflow.com/questions/5114292/break-interrupt-a-time-sleep-in-python
                pass
        else:
            print("Worker cannot receive other workers' updates.")

    def miner_broadcast_updates(self, miners_within_the_same_epoch):

        data = {"miner_id": self._idx, "miner_ip": self._ip_and_port, "received_updates": self._received_transactions}
        headers = {'Content-Type': "application/json"}

        # broadcast the updates
        for miner in miners_within_the_same_epoch:
            response = requests.post(miner + "/receive_updates_from_miner", data=json.dumps(data), headers=headers)
            offline_nodes = set()
            node_online = False
            if response.status_code == 200:
                node_online = True
            else:
                node_online = self.retry_offline_peers(miner)
                requests.post(miner + "/receive_updates_from_miner", data=json.dumps(data), headers=headers)
            if node_online:
                print(f'This miner {self._ip_and_port}({self._idx}) has sent unverified updates to miner {miner}')
            else:
                offline_nodes.add(miner)
        peers.difference_update(offline_nodes)
        return "ok"

    def miner_propagate_the_block(self, block_to_propagate, pow_proof):
        # refresh the miner nodes in case there's any gone offline
        device.find_miners_within_the_same_epoch()

        data = {"miner_id": self._idx, "miner_ip": self._ip_and_port, "propagated_block": block_to_propagate.__dict__, "pow_proof": pow_proof}
        headers = {'Content-Type': "application/json"}

        for miner in self._current_epoch_miner_nodes:
            response = requests.post(miner + "/receive_propagated_block", data=json.dumps(data), headers=headers)
            if response.status_code == 200:
                print(f'This miner {self.get_ip_and_port()}({self._idx}) has sent the propagated block to miner {miner}')
                print('Press ENTER to continue...')

    # TODO THIS FUNCTION MUST ABORT IF RECEIVED A BLOCK FROM ANOTHER MINER!!!
    # if info so few, will it speed up mining? maybe not tho
    # PoX - proof of data, more data to mine, more contribution to the model, more speakability. Problem - dominate. PoM - Mix, one round PoD, and then PoW or PoS(more powerful miner maybe, need careful design of the system based on contribution). Or, based on epoch number to decide what to do? Even, epoch number can be used to decide the difficulty
    def miner_mine_block(self, block_to_mine, starting_nonce):
        if self._is_miner:
            print("Mining the block...")
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
                try:
                    pow_proof, mined_block = self.proof_of_work(block_to_mine, starting_nonce)
                    return pow_proof, mined_block
                except:
                    current_nonce_of_the_candidate_block = self.proof_of_work(block_to_mine, starting_nonce)
                    return current_nonce_of_the_candidate_block
                # propagate the block in main()
                # pdb.set_trace()
                # if DEBUG_MODE:
                #     print(f"miner_mine_block() called, with pow_proof {pow_proof}")
                
            else:
                # will never be called as it's prevented by a outer if condition
                print("No transaction to mine.")
                #TODO Skip or wait and go to the next epoch
                return None
        else:
            print("Worker does not mine transactions.")

def jump_to_download_or_next_epoch_warning():
    global mute_warining
    if mute_warining == False:
        if has_added_propagated_block.is_set():
            print("NOTE: A propagated block has been added. Jump to request worker download.")
        if device.if_jump_to_next_epoch():
            print("NOTE: Chain resynced. Jump to the next epoch.")
    mute_warining = True

# def miner_set_wait_time():
#     if device.is_miner():
#         global miner_accept_updates
#         miner_accept_updates = True
#         print(f"{PROMPT} Miner wait time set to {MINER_WAITING_UPLOADS_TIME}s, waiting for updates...")
#         time.sleep(MINER_WAITING_UPLOADS_TIME)
#         miner_accept_updates = False
#         print(f"{PROMPT} Miner done accepting updates in this epoch.")
#     else:
#         # TODO make return more reasonable
#         return "error"

app = Flask(__name__)

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
    return "True" if miner_waiting_for_uploads.is_set() else "False"

# endpoint to for worker to upload updates to the associated miner
# also miner will remember this worker in self._associated_workers
@app.route('/new_transaction', methods=['POST'])
def new_transaction():
    if miner_waiting_for_uploads.is_set():
        update_data = request.get_json()
        # For SVRG
        # required_fields = ["worker_id", "worker_ip", "local_weight_update", "global_gradients_per_data_point", "computation_time", "this_worker_address"]

        # For linear regression
        required_fields = ["worker_id", "worker_ip", "feature_gradients", "computation_time", "this_worker_address"]
        for field in required_fields:
            if not update_data.get(field):
                return "Invalid transaction(update) data", 404
        device.associate_worker(update_data['this_worker_address'])
        peers.add(update_data['this_worker_address'])
        update_data["tx_received_time"] = time.time()
        device.miner_receive_worker_updates(update_data)
        if DEBUG_MODE:
            print("The received updates: ", update_data)
            # print("Press ENTER to continue...")
    else:
        # though /within_miner_wait_time will return False to stop worker from sending but network delay may make this happen
        return "False"
    return "True"

@app.route('/receive_updates_from_miner', methods=['POST'])
def receive_updates_from_miner():
    sender_miner_id = request.get_json()["miner_id"]
    sender_miner_ip = request.get_json()["miner_ip"]
    with_updates = request.get_json()["received_updates"]
    one_miner_updates = {"from_miner_ip": sender_miner_ip, "from_miner_id": sender_miner_id, "received_updates": with_updates}
    print(f"\nReceived unverified updates from miner {sender_miner_ip}({sender_miner_id}).\n")
    if with_updates:
        device.add_received_updates_from_miners(one_miner_updates)
        print(f"The received broadcasted updates {with_updates}\n")
    else:
        print(f"The received broadcasted updates is empty, which is then tossed.\n")
    print("Press ENTER to continue...")
    return "Success", 200
    
@app.route('/receive_propagated_block', methods=['POST'])
def receive_propagated_block():
    # TODO ABORT THE RUNNING POW!!!
    if accepting_propagated_block.is_set():
        global has_added_propagated_block
        global has_received_propagated_block
        global done_verifying_propagated_block
        if has_added_propagated_block.is_set() or has_received_propagated_block.is_set():
            # TODO in paper it has fork is generated when miner receives a new block, but I think this should be generated when a received propagated block has been added to the chain
            return "Another propagated block is delivered and tossed but should be preserved in a later version."
        has_received_propagated_block.set()
        sender_miner_id = request.get_json()["miner_id"]
        sender_miner_ip = request.get_json()["miner_ip"]
        pow_proof = request.get_json()["pow_proof"]
        propagated_block = request.get_json()["propagated_block"]
        # first verify this block id == chain length
        block_idx = propagated_block["_idx"]
        if int(block_idx) != device.get_blockchain().get_chain_length():
            return "The received propagated block is not sync with this miner's epoch."
        # # check pow proof
        # if pow_proof.startswith('0' * Blockchain.difficulty) and pow_proof == propagated_block.compute_hash(): DONE IN add_block
        # add this block to the chain
        reconstructed_block = Block(block_idx,
                    propagated_block["_transactions"],
                    propagated_block["_block_generation_time"],
                    propagated_block["_previous_hash"],
                    propagated_block["_nonce"],
                    propagated_block['_block_hash'])
        print(f"\nReceived a propagated block from {sender_miner_ip}({sender_miner_id})")
        print(reconstructed_block.__dict__, "\nWith PoW", pow_proof)
        # TODO Still verify dimension or not since this miner may not fully trust the other miner? If not verify, what's the reason?
        if device.add_block(reconstructed_block, pow_proof):
            done_verifying_propagated_block.set()
            # may crash between these two sets? Though odds low
            has_added_propagated_block.set()
            device.set_propagated_block_pow(pow_proof)
            device.propagated_block_has_been_added()
            note_text = "NOTE: A propagated block has been received, verified and added to the this miner's blockchain."
            print('=' * len(note_text))
            print(note_text)
            print('=' * len(note_text))
            print("Press ENTER to continue...")
            return "A propagated block has been mined and added to the blockchain."
        else:
            # should process the next received propagated block
            # has_added_propagated_block.clear()
            done_verifying_propagated_block.set()
            has_received_propagated_block.clear()
            return "The received propagated block is not verified. Mining own block continues."
    else:
        return "Propagated block not accepting by this miner."

''' App Starts Here '''

# pre-defined and agreed fields
# miner use these values to verify data validity
DATA_DIM = 3
SAMPLE_SIZE = 2
# miner waits for 180s to fill its candidate block with updates from devices
MINER_WAITING_UPLOADS_TIME = 10
PROPAGATED_BLOCK_WAITING_TIME = 10
OFFLINE_PEER_RETRY_TIMES = 3
OFFLINE_PEER_WAITING_TIME = 5


PROMPT = ">>>"

# TODO change to False and uncomment miner wait time
# miner_accept_updates = True

# the address to other participating members of the network
# TODO delete offline peers
peers = set()

# create a device with a 4 bytes (8 hex chars) id
# the device's copy of blockchain also initialized
device = Miner(binascii.b2a_hex(os.urandom(4)).decode('utf-8'))

miner_waiting_for_uploads = Event()
has_received_propagated_block = Event()
done_verifying_propagated_block = Event()
has_added_propagated_block = Event()
accepting_propagated_block = Event()

print("Ready to start the node.")


# start the app
# assign tasks based on role
@app.route('/')
def runApp():
    
    global miner_waiting_for_uploads
    global has_added_propagated_block
    global has_received_propagated_block
    global done_verifying_propagated_block
    global accepting_propagated_block
    miner_waiting_for_uploads.clear()
    has_added_propagated_block.clear()
    has_received_propagated_block.clear()
    done_verifying_propagated_block.clear()
    accepting_propagated_block.set()

    print(f"\n==================")
    print(f"|  BlockFL Demo  |")
    print(f"==================\n")
    print(f"Current PoW Difficulty: {Blockchain.difficulty}\n")
    # if DEBUG_MODE:
    #     print("System running in sequential mode...")
       
    # wait for worker maximum wating time

    while True: 
        mute_warining = False
        print(f"Starting epoch {device.get_current_epoch()}...\n")
        print(f"{PROMPT} This is Miner with ID {device.get_idx()}\n")
        #TODO recheck peer validity and remove offline peers
        # hopfully network delay won't cause bug if like the propagated block is added at the moment we clear _has_added_propagated_block to True. Though low odds, still have to think about it
        # if not has_added_propagated_block.is_set():
        # if DEBUG_MODE:
        #     cont = input("First clear all related variables for the new epoch, including all received updates from the last epoch if any and associated workers and miners in order to start a new epoch. Continue?\n")
        # else:
        #     print("NOTE: A propagated block has been added. Jump to request worker download.")
        #     pass

        
        # if not has_added_propagated_block.is_set():
        #     # if DEBUG_MODE:
        #     #     cont = input("Next get all workers in this epoch. Continue?\n")
        #     # get all workers in this epoch, used in miner_receive_worker_updates()
        #     device.get_all_current_epoch_workers()
        # else:
        #     print("NOTE: A propagated block has been added. Jump to request worker download.")
        #     pass
        
        # if not has_added_propagated_block.is_set():
        #     # if DEBUG_MODE:
        #     #     cont = input("Miner is waiting for workers to upload their updates now...\nPress ENTER to continue to the next step if no updates uploaded to this miner to receive the broadcasted updates...\n")
        #         # print("Miner is waiting for workers to upload their updates for 5 secs...")
        #         # time.sleep(5)
            
        #     # waiting for worker's updates. While miner_set_wait_time() is working, miner_receive_worker_updates will check block size by checking and when #(tx) = #(workers), abort the timer 
        #     # TODO uncomment in production miner_set_wait_time()
        # else:
        #     print("NOTE: A propagated block has been added. Jump to request worker download.")
        #     pass

        if not has_added_propagated_block.is_set() and not device.if_jump_to_next_epoch():
            miner_waiting_for_uploads.set()
            print(f"Miner is waiting for workers for {MINER_WAITING_UPLOADS_TIME} sections to upload their updates now...\n")
            waiting_time = MINER_WAITING_UPLOADS_TIME
            while True:
                sys.stdout.write(f'\rWaiting {waiting_time}...')
                time.sleep(1)
                sys.stdout.flush()
                waiting_time -= 1
                if waiting_time == 0:
                    miner_waiting_for_uploads.clear()
                    break
        else:
            jump_to_download_or_next_epoch_warning()
        
        miners_within_the_same_epoch = device.find_miners_within_the_same_epoch()
        if not has_added_propagated_block.is_set() and not device.if_jump_to_next_epoch():
            # miner broadcast received local updates
            #if DEBUG_MODE:
                # cont = input("Next, miners broadcast its received updates to other miners, and in the same time accept the broadcasted updates from other miners as well. Continue?\n")
            print("Now, miners are broadcasting their received updates to other miners.\n")
            device.miner_broadcast_updates(miners_within_the_same_epoch)
        else:
            jump_to_download_or_next_epoch_warning()
        

        # if not has_added_propagated_block.is_set():
        #     # TODO find a better approach to implement, maybe use thread - wait for 180s to receive updates from other miners. Also need to consider about the block size!!
        #     if DEBUG_MODE:
        #         cont = input("Now this miner is waiting to receive the broadcasted updates...\n")
        #     # time.sleep(180)
        #     # start cross-verification
        # else:
        #     print("NOTE: A propagated block has been added. Jump to request worker download.")
        #     pass

        if not has_added_propagated_block.is_set() and not device.if_jump_to_next_epoch():
            # TODO verify uploads? How?
            # if DEBUG_MODE:
                #cont = input("\nNext self and cross verification. Continue?\n")
            print("\nNext self and cross verification.\n")
            candidate_block = device.cross_verification()
        else:
            jump_to_download_or_next_epoch_warning()
        

        if not has_added_propagated_block.is_set() and not device.if_jump_to_next_epoch():
            # miner mine transactions by PoW on this candidate_block
            print("\nNext miner mines its own block.\n")
            if block_to_mine.get_transactions():
                block_starting_nonce = candidate_block.get_current_nonce()
                while True:
                    if not has_received_propagated_block.is_set():
                        try:
                            pow_proof, mined_block = device.miner_mine_block(candidate_block, block_starting_nonce)
                            break
                        except:
                            block_starting_nonce = device.miner_mine_block(candidate_block, block_starting_nonce)
                    else:
                        done_verifying_propagated_block.wait()
                        has_added_propagated_block.wait(2) # afraid of the delay btw these two set()s
                        if has_added_propagated_block.is_set():
                            # the propagated block has been added
                            break
                        else:
                            # TODO check if propagated block added again, afriad that the system crashed btw these two set()s
                            done_verifying_propagated_block.clear()
                            # continue
            else:
                print("No transaction to mine. Wait for 10s for a propagated block...")
                while True:
                    waiting_time = PROPAGATED_BLOCK_WAITING_TIME
                    sys.stdout.write(f'\rWaiting {waiting_time}...')
                    time.sleep(1)
                    sys.stdout.flush()
                    waiting_time -= 1
                    if waiting_time == 0:
                        if has_added_propagated_block.is_set():
                            break
                        else:
                            # resync chain
                            if device.consensus():
                                print("Longer chain is found. For miner, go to next epoch.")
                                self.set_jump_to_next_epoch_True()
                                break
        else:
            jump_to_download_or_next_epoch_warning()
        
        # if not has_added_propagated_block.is_set():
        #     # block_propagation
        #     # TODO if miner_mine_block returns none, which means it gets aborted, then it does not run propagate_the_block and add its own block. If not, run the next two.
        #     if DEBUG_MODE:
        #         cont = input("\nNext miner_propagate_the_block. Continue?\n")
        #     if not has_added_propagated_block.is_set():
        #         device.miner_propagate_the_block(mined_block, pow_proof)
        #     else:
        #         print("NOTE: A propagated block has been added. Jump to request worker download.")
        # else:
        #     print("NOTE: A propagated block has been added. Jump to request worker download.")
        #     pass

        if not has_added_propagated_block.is_set() and not device.if_jump_to_next_epoch():
            # add its own block
            # TODO fork ACK?
            # Now receiving propagated block door closed
            accepting_propagated_block.clear()
            # if DEBUG_MODE:
                # cont = input("\nNext miner adds its own block. Continue?\n")
            print("\nNow miner adds its own block.\n")
            if device.add_block(mined_block, pow_proof):
                print("Its own block has been added.")
            else:
                # TODO
                pass
        else:
            jump_to_download_or_next_epoch_warning()

        # send updates to its associated miners
        if DEBUG_MODE:
            # cont = input("\nNext request_associated_workers_download. Continue?\n")
            print("\nNext request_associated_workers_download.\n")
            if has_added_propagated_block.is_set():
                # download the added propagated block
                device.request_associated_workers_download(device.get_propagated_block_pow())
            else:
                # download its own block
                device.request_associated_workers_download(pow_proof)
            # if DEBUG_MODE:
                # cont = input("Next epoch. Continue?\n")
        # reset vars before starting next epoch
        device.reset_related_vars_for_new_epoch()


# endpoint to return the node's copy of the chain.
# Our application will be using this endpoint to query the contents in the chain to display
@app.route('/chain', methods=['GET'])
def display_chain():
    chain = json.loads(query_blockchain())["chain"]
    print("\nChain info requested and returned -")
    for block_iter in range(len(chain)):
        block_id_to_print = f"Block #{block_iter+1}"
        print()
        print('=' * len(block_id_to_print))
        print(block_id_to_print)
        print('=' * len(block_id_to_print))
        block = chain[block_iter]
        # print("_idx", block["_idx"])
        for tx_iter in range(len(block["_transactions"])):
            print(f"\nTransaction {tx_iter+1}\n", block["_transactions"][tx_iter], "\n")
        print("_block_generation_time", block["_block_generation_time"])
        print("_previous_hash", block["_previous_hash"])
        print("_nonce", block["_nonce"])
        print("_block_hash", block["_block_hash"])
    return "Chain Returned in Port Console"

''' add node to the network '''

# endpoint to add new peers to the network.
# why it's using POST here?
@app.route('/register_node', methods=['POST'])
def register_node():
    pass


@app.route('/register_with', methods=['POST'])
def register_with_existing_node():
    """
    Internally calls the `register_node` endpoint to register current node with the node specified in the
    request, and sync the blockchain as well as peer data.
    """
    pass



''' debug methods '''
@app.route('/get_chain_meta', methods=['GET'])
def query_blockchain():
    pass

@app.route('/get_peers', methods=['GET'])
def query_peers():
    return json.dumps({"peers": list(peers)})

