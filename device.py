# FAULT TOLERANCE
RETRY_CONNECTION_TIMES = 3 #TODO for every response
RETRY_ERROR_RESPONSE = 3 #TODO for every response status code
RETRY_WAITING_TIME_IN_SECONDS = 5

from blockchain import Blockchain
import os
from utils import *

class Device:
	def __init__(self, idx):
		self._idx = idx
		self._is_miner = False
		self._ip_and_port = None
		self._blockchain = Blockchain()
		self._same_epoch_miner_nodes = set()
		# used in cross_verification and in the future PoS
		self._rewards = 0
		self._jump_to_next_epoch = False
		# the address to other participating members of the network
		self._peers = set()

	''' setters '''
	# set the consensused blockchain
	def set_blockchain(self, blockchain):
		self._blockchain = blockchain

	def set_ip_and_port(self, ip_and_port):
		self._ip_and_port = ip_and_port

	def set_jump_to_next_epoch_True(self):
		self._jump_to_next_epoch = True

	def get_rewards(self, rewards):
		self._rewards += rewards

	def add_peers(self, new_peers):
		if type(new_peers) == str:
			self._peers.add(new_peers)
		else:
			self._peers.update(new_peers)
	
	def remove_peers(self, peers_to_remove):
		if type(peers_to_remove) == str:
			self._peers.remove(peers_to_remove)
		else:
			self._peers.difference_update(peers_to_remove)

	def update_miner_nodes(self, new_miner_nodes):
		self._same_epoch_miner_nodes.clear()
		self._same_epoch_miner_nodes.update(new_miner_nodes)

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

	def if_jump_to_next_epoch(self):
		return self._jump_to_next_epoch

	def is_miner(self):
		return self._is_miner

	def get_peers(self):
		return repr(self._peers)

	''' common functions '''
	
	''' system exit in these conditions(with retries already)
	1 - registrar node not found
	2 - no peers found in the network after update_peers
	3 - no miners found for workers
	'''
	# HAS FAULT TOLERANCE
	# REMOVE PEERS ONLY IN HERE
	def update_peers(self):
		offline_peers = set()
		potential_new_peers = set()
		for node in self._peers:
			response_peers = requests.get(f'{node}/get_peers')
			if response_peers.status_code == 200:
				potential_new_peers.update(response_peers.json()['peers'])
			else:
				# FAULT: cannot connect or invalid response
				retry_error_response = RETRY_ERROR_RESPONSE
				while retry_error_response:
					# warning
					print(f"Peer node {node} connection error with status code {response_peers.status_code}. {retry_error_response} reconnecting attempts left...")
					retry_error_response -= 1
					# Wait a gap
					self.wait_to_retry_connection()
					# retry connection
					response_peers = requests.get(f'{node}/get_peers')
					if response_peers.status_code == 200:
						potential_new_peers.update(response_peers.json()['peers'])
						break
					else:
						if retry_error_response == 0:
							# node most likely offline
							print(f"Peer node {node} connection error with status code {response.status_code} after {RETRY_ERROR_RESPONSE} reconnecting attempts. This peer will be removed from the peer list.")
							offline_peers.update(node)
		# https://stackoverflow.com/questions/49348340/how-to-remove-multiple-elements-from-a-set
		self.add_peers(potential_new_peers)
		self.remove_peers(offline_peers)
		# remove itself if there is
		try:
			peers.remove(self.get_ip_and_port())
		except:
			pass
		if not self._peers:
			print("After updating peer list, no peers are found in the network. System aborted. Please restart the node.")
			os._exit(0)

	# FAULT TOLERANCE HELPER FUNCTION 
	def wait_to_retry_connection(self):
		retry_waiting_time_in_seconds = RETRY_WAITING_TIME_IN_SECONDS
		while retry_waiting_time_in_seconds:
			sys.stdout.write(f'\rWaiting {retry_waiting_time_in_seconds}...')
			time.sleep(1)
			sys.stdout.flush()
			retry_waiting_time_in_seconds -= 1

	# FAULT TOLERANCE FOR WORKERS
	# BUT SKIP REMOVING PEERS TO KEEP SIMPLE
	def find_miners_within_the_same_epoch(self):
		retry_error_response = RETRY_ERROR_RESPONSE
		while True:
			self.update_peers()
			miner_nodes = set()
			for node in self._peers:
				response_role = requests.get(f'{node}/get_role')
				if response_role.status_code == 200:
					if response_role.text == 'Miner':
						response_miner = requests.get(f'{node}/get_miner_epoch')
						if response_miner.status_code == 200:
							miner_nodes.add(node)
			self.update_miner_nodes(miner_nodes)
			if self._same_epoch_miner_nodes:
				break
			else:
				if self._is_miner:
					break
				else:
					if retry_error_response:
						print(f"No miners in the same epoch found. {retry_error_response} re-attempts left...")
						print("First try resync chains...")
						if self.pow_consensus():
							# TODO a worker should now do global updates to the point
							print("A longer chain has found. Go to the next epoch.")
							return
						else:
							print("Not a longer chain found. Continue searching for miners.")
						retry_error_response -= 1
						self.wait_to_retry_connection()
						continue
					else:
						print(f"No miners found with the same epoch in the network and not a longer chain has been found after {RETRY_ERROR_RESPONSE} attempts. Miners are essential for workers to operate. Please restart this node and try again.")
						os._exit(0)

	# HAS FAULT TOLERANCE
	def register_with_existing_node(self, registrar_node_address):
		
		data = {"registrant_node_address": self.get_ip_and_port()}
		headers = {'Content-Type': "application/json"}

		retry_connection_times = RETRY_CONNECTION_TIMES
		retry_error_response = RETRY_ERROR_RESPONSE
		while True:
			# Make a request to register with remote a node and sync chain if any
			try:
				response = requests.post(registrar_node_address + "/register_node", data=json.dumps(data), headers=headers)
			except:
				if retry_connection_times:
					print(f"Registrar {registrar_ip_port} not found in the network. {retry_connection_times} re-attempts left...")
					retry_connection_times -= 1
					print(f"Reconnecting in {retry_wait_time} seconds...")
					time.sleep(retry_wait_time)
					continue
				else:
					print(f"Registrar not found in the network after {RETRY_CONNECTION_TIMES} attempts. Please check the registrar address and try again.")
					os._exit(0)
			retry_connection_times = RETRY_CONNECTION_TIMES
			if response.status_code == 200:
				# add registrant node it its peer
				self.add_peers(registrar_node_address)
				# update peer list by registrant node's peer list
				self.add_peers(response.json()['peers'])
				# remove itself if there is
				try:
					self.remove_peers(self.get_ip_and_port())
				except:
					pass
				# sync the chain
				chain_data_dump = response.json()['chain']
				# FAULT: synced chain is invalid
				if not self.sync_chain_from_dump(chain_data_dump):
					# update peers first
					self.update_peers()
					# PoW consensus - set the longest chain in the network as its own chain. If a longer chain is not found, continue to do work
					self.pow_consensus()
					# TODO a worker should now do global updates to the point
				return True
			else:
				# FAULT: invalid response
				if retry_error_response:
					print(f"Registrar node connection error with status code {response.status_code}. {retry_error_response} reconnecting attempts left...")
					retry_error_response -= 1
					self.wait_to_retry_connection()
					continue
				else:
					print(f"Registrar node connection error with status code {response.status_code} after {RETRY_ERROR_RESPONSE} reconnecting attempts. Please restart this node and try again.")
					os._exit(0)

	def register_node(self, registrant_node_address):
		# Add the registrant node to the peer list
		self.add_peers(registrant_node_address)
		print(f"A new registrant {registrant_node_address} has been registered. This node's current peers {self.get_peers()}")
		# Return the consensus blockchain to the new registrant so that the new node can sync
		return self.query_blockchain()

	def query_blockchain(self):
		chain_data = []
		for block in self.get_blockchain()._chain:
			chain_data.append(block.__dict__)
		return json.dumps({"chain_length": len(chain_data),
						"chain": chain_data,
						"peers": list(self._peers)})

	def sync_chain_from_dump(self, chain_dump):
		for block_data in chain_dump:
			block = Block(block_data["_idx"],
						block_data["_transactions"],
						block_data["_block_generation_time"],
						block_data["_previous_hash"],
						block_data["_nonce"])
			pow_proof = block_data['_block_hash']
			# in add_block, check if pow_proof and previous_hash fileds both are valid
			added = self.add_block(block, pow_proof)
			if not added:
				raise Exception("The chain dump is tampered!!")
				return False
		return True

	def add_block(self, block_to_add, pow_proof):
		"""
		A function that adds the block to the chain after two verifications(sanity check).
		"""
		last_block = self._blockchain.get_last_block()
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

			# When syncing by calling consensus(), rebuilt block doesn't have this field. add the block hash after verifying
			block_to_add.set_hash()

			self._blockchain.append_block(block_to_add)
			return True
		else:
			# only check 2. above
			if not self.check_pow_proof(block_to_add, pow_proof):
				return False
			# add genesis block
			block_to_add.set_hash()
			self._blockchain.append_block(block_to_add)
			return True

	def check_pow_proof(block_to_check, pow_proof):
		return pow_proof.startswith('0' * Blockchain.difficulty) and pow_proof == block_to_check.compute_hash()

	def check_chain_validity(self, chain_to_check):
		chain_len = chain_to_check.get_chain_length()
		if chain_len == 0 or chain_len == 1:
			pass
		else:
			for block in chain_to_check[1:]:
				if self.check_pow_proof(block, block.get_block_hash()) and block.get_previous_hash == chain_to_check[chain_to_check.index(block) - 1].compute_hash(hash_previous_block=True):
					pass
				else:
					return False
		return True
	
	# COULD HAVE FAULT TOLERANCE - REMOVING OFFLINE PEERS
	# BUT SKIP REMOVING PEERS TO KEEP SIMPLE
	def pow_consensus(self):
		"""
		PoW consensus algorithm - if a longer valid chain is found, the current device's chain is replaced with it.
		"""
		longest_chain = None
		curr_chain_len = self._blockchain.get_chain_length()

		for node in self._peers:
			response = requests.get(f'{node}/get_chain_meta')
			if response.status_code == 200:
				peer_chain_length = response.json()['chain_length']
				if peer_chain_length > curr_chain_len:
					peer_chain = response.json()['chain']
					if self.check_chain_validity(peer_chain):
						# Longer valid chain found!
						curr_chain_len = peer_chain_length
						longest_chain = peer_chain
			else:
				# simple handling or it would be so complicated
				continue
		if longest_chain:
			self._blockchain._chain = longest_chain
			return True
		return False

	def display_chain(self):
		chain = json.loads(self.query_blockchain())["chain"]
		display_str = "\nChain info requested and returned -\n"
		for block_iter in range(len(chain)):
			block_id_to_print = f"Block #{block_iter+1}"
			display_str += '=' * len(block_id_to_print)
			display_str += block_id_to_print
			display_str += '=' * len(block_id_to_print)
			block = chain[block_iter]
			for tx_iter in range(len(block["_transactions"])):
				display_str += f"\nTransaction {tx_iter+1}\n" + block["_transactions"][tx_iter] + "\n"
			display_str += "_block_generation_time" + block["_block_generation_time"]
			display_str += "_previous_hash" + block["_previous_hash"]
			display_str += "_nonce" + block["_nonce"]
			display_str += "_block_hash" + block["_block_hash"]
		return display_str
 

