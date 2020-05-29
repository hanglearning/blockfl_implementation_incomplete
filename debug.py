import torch

_data = []
_data.append({'x': torch.tensor([[1.0],[2.0]]), 'y': torch.tensor([5.])})
_data.append({'x': torch.tensor([[3.0],[4.0]]), 'y': torch.tensor([11.])})

_data_dim = 2
_sample_size = 2
_step_size = 1

_global_weight_vector = torch.tensor([[0.0],[0.0]])

def worker_local_update_linear_regression():
    local_weight = _global_weight_vector
    feature_gradients_list = torch.zeros(_data_dim, 1)
    for data_point in _data:
        difference_btw_hypothesis_and_true_label = data_point['x'].t()@_global_weight_vector - data_point['y']
        # for feature_value_iter in range(self._data_dim):
        #     feature_gradients_list[feature_value_iter] += (difference_btw_hypothesis_and_true_label * data_point['x'][feature_value_iter]).squeeze(0)
        feature_gradients_list += difference_btw_hypothesis_and_true_label * data_point['x']
    # for feature_gradient_iter in range(_data_dim):
    #     feature_gradients_list[feature_gradient_iter] /= len(_data)
    feature_gradients_list /= len(_data)
    print(feature_gradients_list)

def worker_local_update_SVRG():
    # SVRG algo, BlockFL section II and reference[4] 3.2
    # gradient of loss function chosen - mean squared error
    # delta_fk(wl) for each sk
    global_gradients_per_data_point = []
    # initialize the local weights as the current global weights
    local_weight = _global_weight_vector
    # calculate delta_f(wl)
    # last_block = self._blockchain.get_last_block()
    # if last_block is not None:
    #     transactions = last_block.get_transactions()
    #     ''' transactions = [{'worker_id': 'ddf993e5', 'local_weight_update': {'update_tensor_to_list': [[0.0], [0.0], [0.0], [0.0]], 'tensor_type': 'torch.FloatTensor'}, 'global_gradients_per_data_point': [{'update_tensor_to_list': [[-15.794557571411133], [-9.352561950683594], [-90.67684936523438], [-80.69305419921875]], 'tensor_type': 'torch.FloatTensor'}, {'update_tensor_to_list': [[-132.57232666015625], [-284.4437561035156], [-53.215885162353516], [-13.190389633178711]], 'tensor_type': 'torch.FloatTensor'}, {'update_tensor_to_list': [[-35.0189094543457], [-6.117635250091553], [-23.661569595336914], [-3.7096316814422607]], 'tensor_type': 'torch.FloatTensor'}], 'computation_time': 0.16167688369750977, 'this_worker_address': 'http://localhost:5001', 'tx_received_time': 1587539183.5140128}] '''
    #     tensor_accumulator = torch.zeros_like(self._global_weight_vector)
    #     for update_per_device in transactions:
    #         for data_point_gradient in update_per_device['global_gradients_per_data_point']:
    #             data_point_gradient_list = data_point_gradient['update_tensor_to_list']
    #             data_point_gradient_tensor_type = data_point_gradient['tensor_type']
    #             data_point_gradient_tensor = getattr(torch, data_point_gradient_tensor_type[6:])(data_point_gradient_list)
    #             tensor_accumulator += data_point_gradient_tensor
    #     num_of_device_updates = len(transactions)
    #     delta_f_wl = tensor_accumulator/(num_of_device_updates * self._sample_size)
    # chain is empty now as this is the first epoch. Use its own data sample to accumulate this value
    delta_f_wl = torch.zeros_like(_global_weight_vector)
    for data_point in _data:
        local_weight_track_grad = local_weight.clone().detach().requires_grad_(True)
        fk_wl = (data_point['x'].t()@local_weight_track_grad - data_point['y'])**2/2
        fk_wl.backward()
        delta_f_wl += local_weight_track_grad.grad
    delta_f_wl /= _sample_size
    # ref - https://stackoverflow.com/questions/3620943/measuring-elapsed-time-with-the-time-module
    # start_time = time.time()
    # iterations = the number of data points in a device
    # function(1)
    for data_point_iter in range(len(_data)):
        if data_point_iter == 1:
            print()
        data_point = _data[data_point_iter]
        print(data_point)
        local_weight_track_grad = local_weight.clone().detach().requires_grad_(True)
        # loss of one data point with current local update fk_wil
        
        fk_wil = (data_point['x'].t()@local_weight_track_grad - data_point['y'])**2/2
        # calculate delta_fk_wil
        fk_wil.backward()
        delta_fk_wil = local_weight_track_grad.grad

        last_global_weight_track_grad = _global_weight_vector.clone().detach().requires_grad_(True)
        # loss of one data point with last updated global weights fk_wl
        fk_wl = (data_point['x'].t()@last_global_weight_track_grad - data_point['y'])**2/2
        # calculate delta_fk_wl
        fk_wl.backward()
        delta_fk_wl = last_global_weight_track_grad.grad
        # record this value to upload
        # need to convert delta_fk_wl tensor to list in order to make json.dumps() work
        global_gradients_per_data_point.append({"update_tensor_to_list": delta_fk_wl.tolist(), "tensor_type": delta_fk_wl.type()})
        # pdb.set_trace()
        # calculate local update
        local_weight = local_weight - (_step_size/len(_data)) * (delta_fk_wil - delta_fk_wl + delta_f_wl)
        print(local_weight)
    print(global_gradients_per_data_point)

    # worker_id and worker_ip is not required to be recorded to the block. Just for debugging purpose

worker_local_update_linear_regression()