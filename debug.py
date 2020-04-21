import torch

data_dim = 3

def worker_local_update(self):
    # SVRG algo, BlockFL section II and reference[4] 3.2
    # gradient of loss function chosen - mean squared error
    # delta_fk(wl) for each sk
    global_gradients_per_data_point = []
    # initialize the local weights as the current global weights
    local_weight = torch.zeros(data_dim, 1)
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