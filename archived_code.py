''' Worker '''
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

''' utils.py '''
# NOT USED
def check_candidate_node_address(input_address):
    # https://stackoverflow.com/questions/7160737/python-how-to-validate-a-url-in-python-malformed-or-not
    import re
    regex = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, input_address) is not None