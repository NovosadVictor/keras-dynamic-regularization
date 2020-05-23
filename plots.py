import os
from matplotlib import pyplot as plt
import json


histories = []
with open('histories/histories_05.21.2020_18_58_04.json', 'r') as f:
    histories.append(json.load(f))

dropout_parameters = []
with open('histories/parameters_05.21.2020_18_58_04.json', 'r') as f:
    dropout_parameters.append(json.load(f))


for i in range(len(histories)):
    plt.figure(figsize=(12, 10))
    legends_acc = sum(
        [
            ['{}_train acc'.format(index), '{}_test acc'.format(index)]
            for index in range(len(histories[i]))
        ],
        [],
    )
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    for history in histories[i]:
        # summarize history for accuracy
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])

    for parameters in dropout_parameters[i]:
        plt.plot(parameters)

    plt.legend(legends_acc, loc='upper right')
    plt.show()

    plt.figure(figsize=(12, 10))
    legends_loss = sum(
        [
            ['{}_train loss'.format(index), '{}_test loss'.format(index)]
            for index in range(len(histories[i]))
        ],
        [],
    )
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    for history in histories[i]:
        # summarize history for loss
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])

    for parameters in dropout_parameters[i]:
        plt.plot(parameters)

    plt.legend(legends_loss, loc='upper right')
    plt.show()
