import json
import time
from hashlib import sha256
import torch

class Block:
    def __init__(self, index, transactions, timestamp, previous_hash, nonce=0):
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = nonce

    def compute_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return sha256(block_string.encode()).hexdigest()

class Blockchain:
    difficulty = 2
    def __init__(self):
        self.chain = []
        
    
    # def creat_genesis_block(self):
    #     genesis_block = Block(0,[],0,"0")
    #     genesis_block.hash = genesis_block.compute_hash()
    #     self.chain.append(genesis_block)

    def last_block(self):
        return self.chain[-1]

class Device:
    def __init__(self, global_weight): # what is the initial global_weight?
        self.global_weight = global_weight
        self.data = None
        self.block_size = None            
        self.time_wait = None  #should we set maximum wating time?
        self.unconfirm_transactions = []
        self.blockchain = Blockchain()


    def data_sample(self, data):
        self.data = data

    def set_block_size(self, header_size, model_update_size):         #where are the header size and model update size?
        self.block_size = header_size + model_update_size*len(self.data)

    def local_model_update(self, step_size):
        local_weight = self.global_weight

        start_time = time.time()
		
		for data_point in self.data:

            local_weight = local_weight - (step_size/len(self.data)) * (data_point[0].t()@local_weight - data_point[1])

            self.global_weight[data_iter] = self.global_weight - (step_size/iterations) * ()

        return time.time() - start_time


    def cross_verification(self, t, candidate_block): 
        if len(candidate_block) >= self.block_size
            return False
        else if t >= self.time_wait:
            return True
        else:
            pass

    def proof_of_work(self, block):
        block.nonce = 0

        compute_hash = block.compute_hash()
        while not computed_hash.startswith('0' * self.blockchain.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()

        return computed_hash

    def add_new_transaction(self):
        self.unconfirm_transactions.append(self.global_weight)

    def add_block(self, block, proof):
        """
        A function that adds the block to the chain after verification.
        Verification includes:
        * Checking if the proof is valid.
        * The previous_hash referred in the block and the hash of latest block
          in the chain match.
        """
        previous_hash = self.blockchain.last_block.hash

        if previous_hash != block.previous_hash:
            return False

        if not Device.is_valid_proof(block, proof):
            return False

        block.hash = proof
        self.blockchain.chain.append(block)
        return True

     @classmethod
    def is_valid_proof(cls, block, block_hash):
        """
        Check if block_hash is valid hash of block and satisfies
        the difficulty criteria.
        """
        return (block_hash.startswith('0' * self.blockchain.difficulty) and
                block_hash == block.compute_hash())

    def consensus(self):

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