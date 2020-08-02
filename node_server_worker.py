from device import *
from utils import *
import utils

import pdb

import sys
import random
import time
import torch
import os
import binascii
import copy

from hashlib import sha256


DEBUG_MODE = True # press any key to continue


class Worker(Device):
	def __init__(self, idx):
		Device.__init__(self, idx)
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
		

	''' getters '''

	def get_data(self):
		return self._data

	# may not be used
	def get_global_weight_vector(self):
		return self._global_weight_vector

	''' setters '''
	# set data dimension
	def set_data_dim(self, data_dim):
		self._data_dim = data_dim

	''' Functions for Workers '''

	def reset_related_vars_for_new_epoch(self):
		self._jump_to_next_epoch = False

	# SVRG only
	def worker_set_step_size(self, step_size):
		if step_size <= 0:
			print("Step size has to be positive.")
		else:
			self._step_size = step_size

	def worker_generate_dummy_data(self):
		self._sample_size = random.randint(5, 10)
		# define range
		r1, r2 = 0, 2
		if not self._data:
			self.expected_w = torch.tensor([[3.0], [7.0], [12.0]])
			for _ in range(self._sample_size):
				x_tensor = (r1 - r2) * torch.rand(self._data_dim, 1) + r2
				y_tensor = self.expected_w.t()@x_tensor
				self._data.append({'x': x_tensor, 'y': y_tensor})
			if DEBUG_MODE:
				print(self._data)
		else:
			print("The data of this worker has already been initialized. Changing data is not currently implemented in this version.")

	# worker global weight initialization
	def worker_init_global_weihgt(self):
		self._global_weight_vector = torch.zeros(self._data_dim, 1)
	
	
	def worker_associate_and_upload_to_miner(self, upload):
		self.find_miners_within_the_same_epoch()
		candidate_miners = self._same_epoch_miner_nodes
		if self._is_miner:
			print("Worker does not accept other workers' updates directly")
		else:
			# not necessary to put self.find_miners_within_the_same_epoch() here again because if there are no same epoch miner found in the first try, there won't be any more as a worker will never be faster than a miner. A slow miner will also catch up by pow_consensus to the latest. Thus, pow_consesus will finally let this worker node catch up too. Otherwise, most probably there is no working miner in this network any more.
			while candidate_miners:
				miner_address = random.sample(candidate_miners, 1)[0]
				print(f"This workder {self.get_ip_and_port()}({self.get_idx()}) picks {miner_address} as its associated miner and attempt to upload its updates...")
				candidate_miners.remove(miner_address)
				# print(f"{PROMPT} This workder {self.get_ip_and_port()}({self.get_idx()}) now assigned to miner with address {miner_address}.\n")
				checked = False
				# check again if this node is still a miner 
				response = requests.get(f'{miner_address}/get_role')
				if response.status_code == 200:
					if response.text == 'Miner':
						# check again if worker and miner are in the same epoch
						response_epoch = requests.get(f'{miner_address}/get_miner_epoch')
						if response_epoch.status_code == 200:
							miner_epoch = int(response_epoch.text)
							if miner_epoch == self.get_current_epoch():
								# check if miner is within the wait time of accepting updates
								response_miner_accepting = requests.get(f'{miner_address}/within_miner_wait_time')
								if response_miner_accepting.text == "True":
									checked = True
				if not checked:
					print(f"The picked miner {miner_address} is unavailable. Try resyncing chain first...")
					# first try resync chain
					if self.pow_consensus():
						# TODO a worker should now do global updates to the point
						print("A longer chain has found. Go to the next epoch.")
						return
					else:
						if candidate_miners:
							print("Not a longer chain found. Re-pick another miner and continue...")
							continue
						else:
							print("Most likely there is no miner in the network any more. Please restart this node and try again.")
							os._exists(0)
				else:
					# record this worker's address to let miner request this worker to download the block later
					upload['this_worker_address'] = self._ip_and_port
					# upload
					response_miner_has_accepted = requests.post(
								f"{miner_address}/new_transaction",
								data=json.dumps(upload),
								headers={'Content-type': 'application/json'})
					retry_connection_times = RETRY_CONNECTION_TIMES
					while True:
						if response_miner_has_accepted.text == "True":
							print(f"Upload to miner {miner_address} succeeded!")
							return
						else:
							if retry_connection_times:
								print(f"Upload to miner error. {retry_connection_times} re-attempts left...")
								retry_connection_times -= 1
								# re-upload
								response_miner_has_accepted = requests.post(
								f"{miner_address}/new_transaction",
								data=json.dumps(upload),
								headers={'Content-type': 'application/json'})
							else:
								candidate_miners.remove(miner_address)
								if candidate_miners:
									print(f"Upload to miner error after {RETRY_CONNECTION_TIMES} attempts. Re-pick another miner and continue...")
									break
								else:
									print("Most likely there is no miner in the network any more. Please restart this node and try again.")
									os._exists(0)

	# TODO START FROM HERE, Track rewards on chain
	# def worker_receive_rewards_from_miner(self, rewards):
	#     print(f"Before rewarded, this worker has rewards {self._rewards}.")
	#     self.get_rewards(rewards)
	#     print(f"After rewarded, this worker has rewards {self._rewards}.\n")
	# TODO only make a endpoint to query rewards balance

	def worker_local_update_linear_regresssion(self):
		if self._is_miner:
			print("Miner does not perfrom gradient calculations.")
		else:
			start_time = time.time()
			# https://d18ky98rnyall9.cloudfront.net/_7532aa933df0e5055d163b77102ff2fb_Lecture4.pdf?Expires=1590451200&Signature=QX0rGKTvN6Wc1OgL~M5d23cibJF0fQ7jMWG5dSO3ooaKfYH~Yl4UadTvLQn2KFdUqAMwUaMwKl3kFG65f4w~R62xyumryaHTRDO7K8f5c8kM7v62OYDr0xDvuJ8K3B-Rjr6XbmnCx6tOo6Fi-sAm-fXbWz2cfJVrm6a2jaJU1BI_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A page 8
			# part of the gradient decent formular after alpha*(1/m)
			feature_gradients_tensor = torch.zeros(self._data_dim, 1)
			for data_point in self._data:
				difference_btw_hypothesis_and_true_label = data_point['x'].t()@self._global_weight_vector - data_point['y']
				feature_gradients_tensor += difference_btw_hypothesis_and_true_label * data_point['x']
			feature_gradients_tensor /= len(self._data)
		
		print(f"Current global_weights: {self._global_weight_vector}")
		print(f"Abs difference from expected weights({self.expected_w}): {abs(self.expected_w - self._global_weight_vector)}")
		return {"worker_id": self._idx, "worker_ip": self._ip_and_port, "feature_gradients": {"feature_gradients_list": feature_gradients_tensor.tolist(), "tensor_type": feature_gradients_tensor.type()}, "computation_time": time.time() - start_time}

	def linear_regression_one_epoch(self, transaction):
		# alpha
		learning_rate = 0.1
		feature_gradients_tensor_accumulator = torch.zeros_like(self._global_weight_vector)
		num_of_device_updates = 0
		for update in transaction:
			num_of_device_updates += 1
			feature_gradients_list = update["feature_gradients"]["feature_gradients_list"]
			feature_gradients_tensor_type = update["feature_gradients"]["tensor_type"]
			feature_gradients_tensor = getattr(torch, feature_gradients_tensor_type[6:])(feature_gradients_list)
			feature_gradients_tensor_accumulator += feature_gradients_tensor
		# perform global updates by gradient decent
		self._global_weight_vector -= learning_rate * feature_gradients_tensor_accumulator/num_of_device_updates
		print('updated self._global_weight_vector', self._global_weight_vector)
		print('abs difference from expected weights', abs(self._global_weight_vector - self.expected_w))

		with open(f'/Users/chenhang91/TEMP/Blockchain Research/convergence_logs/updated_weights_{self._idx}.txt', "a") as myfile:
			myfile.write(str(self._global_weight_vector)+'\n')
		with open(f'/Users/chenhang91/TEMP/Blockchain Research/convergence_logs/weights_diff_{self._idx}.txt', "a") as myfile:
			myfile.write(str(abs(self._global_weight_vector - self.expected_w))+'\n')

		print()
		for data_point_iter in range(len(self._data)):
			data_point = self._data[data_point_iter]
			print(f"For datapoint {data_point_iter}, abs difference from true label: {abs(self._global_weight_vector.t()@data_point['x']-data_point['y'])}")
			with open(f'/Users/chenhang91/TEMP/Blockchain Research/convergence_logs/prediction_diff_point_{self._idx}_{data_point_iter+1}.txt', "a") as myfile:
				myfile.write(str(abs(self._global_weight_vector.t()@data_point['x']-data_point['y']))+'\n')


	def worker_global_update_linear_regression(self):
		print("This worker is performing global updates...")
		transactions_in_downloaded_block = self._blockchain.get_last_block().get_transactions()
		print("transactions_in_downloaded_block", transactions_in_downloaded_block)
		self.linear_regression_one_epoch(transactions_in_downloaded_block)
		print("====================")
		print("Global Update Done.")
		# print("Press ENTER to continue to the next epoch...")

	def post_resync_linear_regression(self):
		self.worker_init_global_weihgt()
		newly_synced_blockchain = self.get_blockchain()
		# overwrite log files
		open(f'/Users/chenhang91/TEMP/Blockchain Research/convergence_logs/updated_weights_{self._idx}.txt', "w")
		open(f'/Users/chenhang91/TEMP/Blockchain Research/convergence_logs/weights_diff_{self._idx}.txt', "w")
		for data_point_iter in range(len(self._data)):
			open(f'/Users/chenhang91/TEMP/Blockchain Research/convergence_logs/prediction_diff_point_{self._idx}_{data_point_iter+1}.txt', "w")
		for block in device.get_blockchain()._chain:
			transactions_in_block = block.get_transactions()
			self.linear_regression_one_epoch(block)

""" End Point Functions """

''' add node to the network '''

# TODO update peers from its miner at every round?

# endpoint to add new peers to the network.
@app.route('/register_node', methods=['POST'])
def register_node():
	registrant_node_address = request.get_json()["registrant_node_address"]
	if not registrant_node_address:
		return "Invalid data", 400
	return device.register_node(registrant_node_address)


@app.route('/register_with', methods=['POST'])
def register_with_existing_node():
	"""
	Calls the `register_node` endpoint at the registrar to register current node, and sync the blockchain as well as peer data from the registrar.
	"""
	registrar_node_address = request.get_json()["registrar_node_address"]
	if not registrar_node_address:
		return "Invalid data", 400
	if device.register_with_existing_node(registrar_node_address):
		print(f"Node {device.get_ip_and_port()} registered with {registrar_node_address}. This node's current peer list is {device.get_peers()}")
		return "Success", 201

''' query/debug data '''

@app.route('/running', methods=['GET'])
def running():
	return "running", 888

@app.route('/get_role', methods=['GET'])
def return_role():
	return "Miner" if device.is_miner() else "Worker"

@app.route('/get_worker_data', methods=['GET'])
def return_data():
	json.dumps({"data": device.get_data()})

# used while miner check for block size in miner_receive_worker_updates()
@app.route('/get_worker_epoch', methods=['GET'])
def get_worker_epoch():
	if not device.is_miner():
		return str(device.get_current_epoch())
	else:
		# TODO make return more reasonable
		return "error"

@app.route('/get_chain_meta', methods=['GET'])
def query_blockchain():
	pass

@app.route('/get_peers', methods=['GET'])
def query_peers():
	return json.dumps({"peers": list(device.get_peers())})

# endpoint to return the node's copy of the chain.
@app.route('/chain', methods=['GET'])
def display_chain():
	device.display_chain()
	return "Chain Returned"


''' App Starts Here '''

# pre-defined and agreed fields
DATA_DIM = 3 # MUST BE CONSISTENT ACROSS ALL WORKERS
STEP_SIZE = 1
EPSILON = 0.02
GLOBAL_BLOCK_AND_UPDATE_WAITING_TIME = 10

PROMPT = ">>>"

# create a worker with a 4 bytes (8 hex chars) id
# the device's copy of blockchain also initialized
device = Worker(binascii.b2a_hex(os.urandom(4)).decode('utf-8'))

def main():
	ip, port, registrar_ip_port = utils.parse_commands()
	this_node_address = f"http://{ip}:{port}"
	device.set_ip_and_port(this_node_address)
	utils.start_flask_app(ip, port, this_node_address, registrar_ip_port)

# https://stackoverflow.com/questions/25029537/interrupt-function-execution-from-another-function-in-python
# https://www.youtube.com/watch?v=YSjIisKdgD0
global_update_or_chain_resync_done = Event()

# start the app
# assign tasks based on role
@app.route('/')
def runApp():
	#TODO recheck peer validity and remove offline peers
	print(f"\n==================")
	print(f"|  BlockFL Demo  |")
	print(f"==================\n")
	
	print(f"{PROMPT} Device is setting data dimensionality {DATA_DIM}")
	device.set_data_dim(DATA_DIM)
	print(f"{PROMPT} Step size set to {STEP_SIZE}")
	device.worker_set_step_size(STEP_SIZE)
	print(f"{PROMPT} Worker sets global_weight to all 0s.")
	device.worker_init_global_weihgt()
	print(f"{PROMPT} Device is generating the dummy data.\n")
	device.worker_generate_dummy_data()
	print(f"Dummy data generated.")

	# while registering, chain was synced, if any
	# TODO change to < EPSILON
	epochs = 0
	while epochs < 150: 
		print(f"\nStarting epoch {device.get_current_epoch()}...\n")
		print(f"{PROMPT} This is workder with ID {device.get_idx()}")
		print("\nStep1. Worker is performing local gradients calculation...\n")
		# if DEBUG_MODE:
			# cont = input("\nStep1. first let worker do local updates. Continue?\n")
		upload = device.worker_local_update_linear_regresssion()
		print("Local updates done.")
		# used for debugging
		if DEBUG_MODE:
			# for SVRG
			# print(f"local_weight_update: {upload['local_weight_update']}")
			# print(f"global_gradients_per_data_point: {upload['global_gradients_per_data_point']}")
			print(f"feature_gradients: {upload['feature_gradients']}")
			print(f"computation_time: {upload['computation_time']}")
		# worker associating with miner
		# if DEBUG_MODE:
			# cont = input("\nStep2. Now, worker will associate with a miner in its peer list and upload its updates to this miner. Continue?\n")
		print("\nStep2. Now, worker will associate with a miner in its peer list and upload its updates to this miner.\n")
		miners_list = device.find_miners_within_the_same_epoch()
		# if DEBUG_MODE:
		#     print("miner_address", miner_address)
		if miners_list is not None:
			# if miners_list is None, meaning the chain is resynced
			# worker uploads data to miner
			device.worker_associate_and_upload_to_miner(upload, miners_list)
		# else: dealt with after combining two classes
		#     wait_new_miner_time = 10
		#     print(f"No miner in peers yet. Re-requesting miner address in {wait_new_miner_time} secs")
		#     time.sleep(wait_new_miner_time)
		#     miner_address = device.find_miners_within_the_same_epoch()
		# TODO during this time period the miner may request the worker to download the block and finish global updating. Need thread programming!
		#if DEBUG_MODE:
			# https://stackoverflow.com/questions/517127/how-do-i-write-output-in-same-place-on-the-console
		if not device.if_jump_to_next_epoch():
			print(f"Now, worker is waiting for {GLOBAL_BLOCK_AND_UPDATE_WAITING_TIME}s to download the added block from its associated miner to do global updates...\n")
			global global_update_or_chain_resync_done
			while True:
				waiting_time = GLOBAL_BLOCK_AND_UPDATE_WAITING_TIME
				while not global_update_or_chain_resync_done.is_set():
					sys.stdout.write(f'\rWaiting {waiting_time}...')
					time.sleep(1)
					sys.stdout.flush()
					waiting_time -= 1
					if waiting_time == 0:
						break
				# global_update_or_chain_resync_done is set() in /download_block_from_miner
				if global_update_or_chain_resync_done.is_set():
					# begin next epoch
					global_update_or_chain_resync_done.clear()
					break
				else:
					# resync chain to see if a longer chain can be found
					if device.consensus():
						print("Longer chain is found. Recalculating global model...")
						self.post_resync_linear_regression()
						global_update_or_chain_resync_done.clear()
						break
					else:
						# not found a longer chain, go back to wait for the download
						print("\nResetting waiting for global update timer...")
						pass

			# print("Now, worker is waiting to download the added block from its associated miners to do global updates for 5 secs...")
		# adjust based on difficulty... Maybe not limit this. Accept at any time. Then give a fork ark. Set a True flag.
		# time.sleep(180)
		# if DEBUG_MODE:
		#     cont = input("Next epoch. Continue?\n")
		device.reset_related_vars_for_new_epoch()
		epochs += 1

@app.route('/get_rewards_from_miner', methods=['POST'])
def get_rewards_from_miner():
	received_rewards = request.get_json()['rewards']
	print(f"\nThis worker received self verification rewards {received_rewards} from the associated miner {request.get_json()['miner_ip']}({request.get_json()['miner_id']})")
	device.worker_receive_rewards_from_miner(received_rewards)
	return "Success", 200

		
@app.route('/download_block_from_miner', methods=['POST'])
def download_block_from_miner():
	print(f"\nReceived downloaded block from the associated miner {request.get_json()['miner_ip']}({request.get_json()['miner_id']})")
	downloaded_block = request.get_json()["block_to_download"]
	pow_proof = request.get_json()["pow_proof"]
	# rebuild the block
	rebuilt_downloaded_block = Block(downloaded_block["_idx"],
					  downloaded_block["_transactions"],
					  downloaded_block["_block_generation_time"],
					  downloaded_block["_previous_hash"],
					  downloaded_block["_nonce"])

	added = device.worker_add_block(rebuilt_downloaded_block, pow_proof)
	# TODO proper way to trigger global update??
	if added:
		# device.worker_global_update_SVRG()
		device.worker_global_update_linear_regression()
		global global_update_or_chain_resync_done
		global_update_or_chain_resync_done.set()
		return "Success", 201
	else:
		# The downloaded block might have been damped. Resync chain
		while True:
			if device.consensus():
				self.post_resync_linear_regression()
				return "Chain Resynced"
			else:
				pass

if __name__ == "__main__":
	main()