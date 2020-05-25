import os
from matplotlib import pyplot as plt
import json

histories_dir = 'histories'

history_files = [name for name in os.listdir(histories_dir) if 'history' in name]
parameter_files = [name for name in os.listdir(histories_dir) if 'parameters' in name]

histories = []
for name in history_files:
    with open(os.path.join(histories_dir, name), 'r') as f:
        histories.append(json.load(f))

parameters = []
for name in parameter_files:
    with open(os.path.join(histories_dir, name), 'r') as f:
        parameters.append([float(name.split('_')[4])] + json.load(f))


plt.figure(figsize=(14, 10))
legends_acc = sum(
    [
        ['{}_test acc'.format(history_files[i])]
        for i in range(len(histories))
    ],
    [],
)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
for i in range(len(histories)):
    # summarize history for accuracy
    color = 'red'
    if 'True' in history_files[i]:
        color = 'blue'
    plt.plot(histories[i]['val_accuracy'])

    # plt.plot([j * 5 for j in range(len(parameters[i]))], parameters[i])

plt.ylim([0.75, 0.9])
plt.legend(legends_acc, loc='lower right')
plt.show()
