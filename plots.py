import os
from matplotlib import pyplot as plt
import json

histories_dir = 'histories/overfitting/'

history_files = [name for name in os.listdir(histories_dir) if 'history' in name]
parameter_files = [name for name in os.listdir(histories_dir) if 'parameters' in name]

histories = []
for name in history_files:
    with open(os.path.join(histories_dir, name), 'r') as f:
        histories.append(json.load(f))

parameters = []
for name in parameter_files:
    with open(os.path.join(histories_dir, name), 'r') as f:
        params = json.load(f)
        if params:
            params = [float(name.split('_')[4])] + params
        parameters.append(params)


plt.figure(figsize=(14, 10))
legends_acc = sum(
    [
        ['{}_test acc'.format(history_files[i])]
        for i in range(len(histories))
    ],
    [],
)
plt.title('dropout')
plt.ylabel('dropout')
plt.xlabel('epoch')
for i in range(len(histories)):
    # summarize history for accuracy
    color = 'red'
    if 'True' in history_files[i]:
        color = 'blue'
    if 'None' in history_files[i]:
        color = 'green'
    # if 'True' in history_files[i]:
    plt.plot(histories[i]['val_accuracy'], color=color)
    plt.plot(histories[i]['accuracy'], color=color)

    # if 'True' in history_files[i]:
    plt.plot([j * 5 for j in range(len(parameters[i]))], parameters[i], color=color)

# plt.ylim([0.7, 1])
plt.legend(legends_acc, loc='lower right')
plt.show()
