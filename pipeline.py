import json
from datetime import datetime
from matplotlib import pyplot as plt

import tensorflow as tf

from constants.utils import load_tf_dataset, prettify_datetime
from constants.classes.model import CNNModel

# device_name = tf.test.gpu_device_name()
# print(device_name)

(ds_train, ds_test), ds_info, input_shape = load_tf_dataset('cifar10')
n_classes = ds_info.features['label'].num_classes
print(n_classes)

histories = []
# with tf.device(device_name):
dropout_parameters = []
for is_dynamic_dropout in [False, True]:
    model = CNNModel(
        input_shape=input_shape,
        n_classes=n_classes,
        is_dynamic_dropout=is_dynamic_dropout,
    )
    model.fit(
        ds_train,
        n_epochs=5,
        validation_data=ds_test,
        is_show=False,
    )

    score = model.model.evaluate(ds_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    histories.append(model.history)
    dropout_parameters.append(model.parameters)


with open('histories/histories_{}.json'.format(prettify_datetime(datetime.now())), 'w') as histories_file:
    json.dump(histories, histories_file, indent=4)
with open('histories/parameters_{}.json'.format(prettify_datetime(datetime.now())), 'w') as parameters_file:
    json.dump(dropout_parameters, parameters_file, indent=4)

plt.figure(figsize=(12, 10))
legends_acc = sum(
    [
        ['{}_train acc'.format(index), '{}_test acc'.format(index)]
        for index in range(len(histories))
    ],
    [],
)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

for history in histories:
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])

for parameters in dropout_parameters:
    plt.plot(parameters)
plt.legend(legends_acc, loc='upper right')
plt.show()

plt.figure(figsize=(12, 10))
legends_loss = sum(
    [
        ['{}_train loss'.format(index), '{}_test loss'.format(index)]
        for index in range(len(histories))
    ],
    [],
)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

for history in histories:
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])

for parameters in dropout_parameters:
    plt.plot(parameters)

plt.legend(legends_loss, loc='upper right')
plt.show()
