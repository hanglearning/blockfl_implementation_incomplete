# https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
# https://stackabuse.com/read-a-file-line-by-line-in-python/
# https://medium.com/@allwindicaprio/plotting-graphs-using-python-and-matplotlib-f55c9b99c338
# https://kite.com/python/answers/how-to-plot-multiple-lines-on-the-same-graph-in-matplotlib-in-python
# https://intellipaat.com/community/8780/append-integer-to-beginning-of-list-in-python
import re
import matplotlib.pyplot as plt

WEIGHTS_NUM = 3
WEIGHT_INIT_VAL = 0.0
EXPECTED_WEIGHTS = [3.0, 7.0, 12.0]
DEVICES_NAMES = ['9f6c88db', '4287aa0e', 'c4415f60']
DEVICES_POINTS_NUM = [10, 7, 6]

# plot updated weights
if False:
    filepath = f'/Users/chenhang91/TEMP/Blockchain Research/convergence_logs/updated_weights_{DEVICES_NAMES[0]}.txt'
    for i in range(1, WEIGHTS_NUM+1):
        vars()[f'weight{i}'] = [0.0]
    with open(filepath) as fp:
        for i in range(1, WEIGHTS_NUM+1):
            vars()[f'line{i}'] = fp.readline()
        while vars()[f'line{WEIGHTS_NUM}']:
            for i in range(1, WEIGHTS_NUM+1):
                vars()[f'weight{i}'].append(float(re.findall("\d+\.\d+", vars()[f'line{i}'])[0]))
                vars()[f'line{i}'] = fp.readline()

    x_cor = range(len(weight1))
    for i in range(1, WEIGHTS_NUM+1):
        plt.plot(x_cor, vars()[f'weight{i}'], label=f'weight{i}')
        plt.plot(x_cor, len(weight1)*[EXPECTED_WEIGHTS[i-1]], linestyle='dashed', label=f'expected_weight{i}')

    plt.legend(loc='best')
    plt.xlabel('epoch number')
    plt.ylabel('updated weights value')

    plt.title('Weights updates vs Epochs')

    plt.show()



# plot weights difference
if False:
    filepath = f'/Users/chenhang91/TEMP/Blockchain Research/convergence_logs/weights_diff_{DEVICES_NAMES[0]}.txt'
    for i in range(1, WEIGHTS_NUM+1):
        vars()[f'weight_diff_{i}'] = [EXPECTED_WEIGHTS[i-1]]

    with open(filepath) as fp:
        for i in range(1, WEIGHTS_NUM+1):
            vars()[f'line{i}'] = fp.readline()
        while vars()[f'line{WEIGHTS_NUM}']:
            for i in range(1, WEIGHTS_NUM+1):
                vars()[f'weight_diff_{i}'].append(float(re.findall("\d+\.\d+", vars()[f'line{i}'])[0]))
                vars()[f'line{i}'] = fp.readline()

    x_cor = range(len(weight_diff_1))
    for i in range(1, WEIGHTS_NUM+1):
        plt.plot(x_cor, vars()[f'weight_diff_{i}'], label=f'weight_diff_{i}')

    plt.legend(loc='best')
    plt.xlabel('epoch number')
    plt.ylabel('difference between updated weights and expected weights')

    plt.title('Weights differences vs Epochs')

    plt.show()

# plot difference for device i(one graph) with their points j(belong to i)
for device_name_iter in range(len(DEVICES_NAMES)):
    device_name = DEVICES_NAMES[device_name_iter]
    device_points_num = DEVICES_POINTS_NUM[device_name_iter]
    for point_num in range(device_points_num):
        filepath = f'/Users/chenhang91/TEMP/Blockchain Research/convergence_logs/prediction_diff_point_{device_name}_{point_num+1}.txt'
        vars()[f'pre_diff_{device_name}_{point_num+1}'] = []
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                vars()[f'pre_diff_{device_name}_{point_num+1}'].append(float(re.findall("\d+\.\d+", line)[0]))
                line = fp.readline()
    x_cor = range(len(vars()[f'pre_diff_{device_name}_1']))
    for point_num_iter in range(device_points_num):
        plt.plot(x_cor, vars()[f'pre_diff_{device_name}_{point_num_iter+1}'], label=f'pre_diff_point_{point_num_iter+1}')
    plt.legend(loc='best')
    plt.xlabel('epoch number')
    plt.ylabel('difference between predicted value and real values')

    plt.title(f'Device {device_name_iter+1} {device_name} Prediction differences vs Epochs')
    plt.show()
