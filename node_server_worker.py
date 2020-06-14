import pdb

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


from block import Block
from blockchain import Blockchain
from device import Device
from device import *

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

    # get global_weight_vector, used while being the register_with node to sync with the registerer node
    def get_global_weight_vector(self):
        return self._global_weight_vector

    ''' setters '''
    # set data dimension
    def set_data_dim(self, data_dim):
        self._data_dim = data_dim

    ''' Functions for Workers '''

    def reset_related_vars_for_new_epoch(self):
        self._jump_to_next_epoch = False

    def worker_set_sample_size(self, sample_size):
        self._sample_size = sample_size

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
    
    
    def worker_associate_and_upload_to_miner(self, upload, miners_list):
        if self._is_miner:
            print("Worker does not accept other workers' updates directly")
        else:
            while True:
                miner_address = random.sample(miners_list, 1)[0]
                print(f"{PROMPT} This workder {device.get_ip_and_port()}({device.get_idx()}) now assigned to miner with address {miner_address}.\n")
                checked = False
                # check if this node is still a miner 
                response = requests.get(f'{miner_address}/get_role')
                if response.status_code == 200:
                    if response.text == 'Miner':
                        # check if worker and miner are in the same epoch
                        response_epoch = requests.get(f'{miner_address}/get_miner_epoch')
                        if response_epoch.status_code == 200:
                            miner_epoch = int(response_epoch.text)
                            if miner_epoch == self.get_current_epoch():
                                checked = True
                            else:
                                pass
                                # TODO not performing the same epoch, resync the chain
                                # consensus()?
                if checked:
                    # check if miner is within the wait time of accepting updates
                    response_miner_accepting = requests.get(f'{miner_address}/within_miner_wait_time')
                    if response_miner_accepting.status_code == 200:
                        if response_miner_accepting.text == "True":
                            # send this worker's address to let miner remember to request this worker to download the block later
                            upload['this_worker_address'] = self._ip_and_port
                            miner_upload_endpoint = f"{miner_address}/new_transaction"
                            #miner_upload_endpoint = "http://127.0.0.1:5001/new_transaction"
                            response_miner_has_accepted = requests.post(miner_upload_endpoint,
                                data=json.dumps(upload),
                                headers={'Content-type': 'application/json'})
                            if response_miner_has_accepted.text == "True":
                                return
                            else:
                                pass
                        else:
                            # # TODO What to do next?
                            # return "Not within miner waiting time."
                            # reassign a miner
                            pass
                    else:
                        pass
                        # return "Error getting miner waiting status", response_miner_accepting.status_code
                else:
                    # first try resync chain
                    if device.consensus():
                        print("Longer chain is found. Recalculating global model...")
                        self.post_resync_linear_regression()
                        self.set_jump_to_next_epoch_True()
                        return
                    # reassign a miner
                    miners_list.remove(miner_address)
                    miner_address = random.sample(miners_list, 1)[0]
    
    def worker_receive_rewards_from_miner(self, rewards):
        print(f"Before rewarded, this worker has rewards {self._rewards}.")
        self.get_rewards(rewards)
        print(f"After rewarded, this worker has rewards {self._rewards}.\n")

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

    # BlockFL step 1 - train with regression
    # return local computation time, and delta_fk(wl) as a list
    # global_gradient is calculated after updating the global_weights
    def worker_local_update_SVRG(self):
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
            # pdb.set_trace()
            last_block = self._blockchain.get_last_block()
            if last_block is not None:
                transactions = last_block.get_transactions()
                ''' transactions = [{'worker_id': 'ddf993e5', 'local_weight_update': {'update_tensor_to_list': [[0.0], [0.0], [0.0], [0.0]], 'tensor_type': 'torch.FloatTensor'}, 'global_gradients_per_data_point': [{'update_tensor_to_list': [[-15.794557571411133], [-9.352561950683594], [-90.67684936523438], [-80.69305419921875]], 'tensor_type': 'torch.FloatTensor'}, {'update_tensor_to_list': [[-132.57232666015625], [-284.4437561035156], [-53.215885162353516], [-13.190389633178711]], 'tensor_type': 'torch.FloatTensor'}, {'update_tensor_to_list': [[-35.0189094543457], [-6.117635250091553], [-23.661569595336914], [-3.7096316814422607]], 'tensor_type': 'torch.FloatTensor'}], 'computation_time': 0.16167688369750977, 'this_worker_address': 'http://localhost:5001', 'tx_received_time': 1587539183.5140128}] '''
                tensor_accumulator = torch.zeros_like(self._global_weight_vector)
                for update_per_device in transactions:
                    for data_point_gradient in update_per_device['global_gradients_per_data_point']:
                        data_point_gradient_list = data_point_gradient['update_tensor_to_list']
                        data_point_gradient_tensor_type = data_point_gradient['tensor_type']
                        data_point_gradient_tensor = getattr(torch, data_point_gradient_tensor_type[6:])(data_point_gradient_list)
                        tensor_accumulator += data_point_gradient_tensor
                num_of_device_updates = len(transactions)
                delta_f_wl = tensor_accumulator/(num_of_device_updates * self._sample_size)
            else:
                # chain is empty now as this is the first epoch. Use its own data sample to accumulate this value
                delta_f_wl = torch.zeros_like(self._global_weight_vector)
                for data_point in self._data:
                    local_weight_track_grad = local_weight.clone().detach().requires_grad_(True)
                    fk_wl = (data_point['x'].t()@local_weight_track_grad - data_point['y'])**2/2
                    fk_wl.backward()
                    delta_f_wl += local_weight_track_grad.grad
                delta_f_wl /= self._sample_size
            # ref - https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module
            start_time = time.time()
            # iterations = the number of data points in a device
            # function(1)
            for data_point in self._data:
                local_weight_track_grad = local_weight.clone().detach().requires_grad_(True)
                # loss of one data point with current local update fk_wil
                
                fk_wil = (data_point['x'].t()@local_weight_track_grad - data_point['y'])**2/2
                # calculate delta_fk_wil
                fk_wil.backward()
                delta_fk_wil = local_weight_track_grad.grad

                last_global_weight_track_grad = self._global_weight_vector.clone().detach().requires_grad_(True)
                # loss of one data point with last updated global weights fk_wl
                fk_wl = (data_point['x'].t()@last_global_weight_track_grad - data_point['y'])**2/2
                # calculate delta_fk_wl
                fk_wl.backward()
                delta_fk_wl = last_global_weight_track_grad.grad
                # record this value to upload
                # need to convert delta_fk_wl tensor to list in order to make json.dumps() work
                global_gradients_per_data_point.append({"update_tensor_to_list": delta_fk_wl.tolist(), "tensor_type": delta_fk_wl.type()})
                # calculate local update
                local_weight = local_weight - (self._step_size/len(self._data)) * (delta_fk_wil - delta_fk_wl + delta_f_wl)

            # worker_id and worker_ip is not required to be recorded to the block. Just for debugging purpose
            return {"worker_id": self._idx, "worker_ip": self._ip_and_port, "local_weight_update": {"update_tensor_to_list": local_weight.tolist(), "tensor_type": local_weight.type()}, "global_gradients_per_data_point": global_gradients_per_data_point, "computation_time": time.time() - start_time}

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

    # TODO
    def worker_global_update_SVRG(self):
        print("This worker is performing global updates...")
        transactions_in_downloaded_block = self._blockchain.get_last_block().get_transactions()
        print("transactions_in_downloaded_block", transactions_in_downloaded_block)
        Ni = self._sample_size
        Nd = len(transactions_in_downloaded_block)
        Ns = Nd * Ni
        global_weight_tensor_accumulator = torch.zeros_like(self._global_weight_vector)
        for update in transactions_in_downloaded_block:
            # convert list to tensor 
            # https://www.aiworkbox.com/lessons/convert-list-to-pytorch-tensor
            # Call function by function name
            # https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string
            updated_weigts_list = update["local_weight_update"]["update_tensor_to_list"]
            updated_weigts_tensor_type = update["local_weight_update"]["tensor_type"]
            updated_weigts_tensor = getattr(torch, updated_weigts_tensor_type[6:])(updated_weigts_list)
            print("updated_weigts_tensor", updated_weigts_tensor)
            global_weight_tensor_accumulator += (Ni/Ns)*(updated_weigts_tensor - self._global_weight_vector)
        self._global_weight_vector += global_weight_tensor_accumulator
        print('self._global_weight_vector', self._global_weight_vector)
        print("Global Update Done.")
        print("Press ENTER to continue to the next epoch...")


''' App Starts Here '''


# pre-defined and agreed fields
DATA_DIM = 3 # MUST BE CONSISTENT ACROSS ALL WORKERS
STEP_SIZE = 1
EPSILON = 0.02
GLOBAL_BLOCK_AND_UPDATE_WAITING_TIME = 10
OFFLINE_PEER_RETRY_TIMES = 3

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
    if not device.is_miner():
        return str(device.get_current_epoch())
    else:
        # TODO make return more reasonable
        return "error"

# https://stackoverflow.com/questions/25029537/interrupt-function-execution-from-another-function-in-python
# https://www.youtube.com/watch?v=YSjIisKdgD0
global_update_or_chain_resync_done = Event()


# print("Please input a host ip for this node to run on. Example format: 0.0.0.0")
# run_on_ip = input("Directly press enter to skip and use localhost: ")
# while not check_ip_format(run_on_ip):
#     run_on_ip = input("IP format error. Please try again: ")
# run_on_ip = "0.0.0.0" if run_on_ip == '' else run_on_ip
# run_on_port = input("Directly press enter to skip and use localhost: ")


# register_with_node = input("\nPlease input a peer address with port number by the example format - http://127.0.0.1:5000, to register with a node in the network.\nIf this is the first node in the network, press to skip this step...")
# if device.check_candidate_node_address(register_with_node):
#     print(f"Registering with node {register_with_node}...")
#     response = requests.post(f'{register_with_node}/register_with', data=f'{"register_with_node_address": "http://127.0.0.1:5000"}', headers=headers)
# else:
#     print(f"Skip registering. Node is ready to start.")

# print("Ready to start the node.")
def main():
    ip, port, registerer_ip_port = parse_commands()
    device.set_ip_and_port(f"{ip}:{port}")
    # register with peer
    if registerer_ip_port:
        register
    # run app
    # https://kite.com/python/examples/4348/flask-get-the-ip-address-of-a-request
    app.run(host=ip, port=port)
    



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


@app.route('/get_chain_meta', methods=['GET'])
def query_blockchain():
    pass

@app.route('/get_peers', methods=['GET'])
def query_peers():
    return json.dumps({"peers": list(peers)})


''' add node to the network '''

# TODO update peers from its miner at every round?

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


if __name__ == "__main__":
	main()