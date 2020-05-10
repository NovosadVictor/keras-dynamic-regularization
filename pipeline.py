from matplotlib import pyplot as plt

from constants.utils import \
    load_mnist
from constants.classes.model import CNNModel

n_classes = 10
img_rows, img_cols = 28, 28

input_shape, x_train, y_train, x_test, y_test = load_mnist(img_rows, img_cols, n_classes)
histories = []
for is_dynamic_dropout in [False, True]:
    model = CNNModel(
        input_shape=input_shape,
        n_classes=n_classes,
        is_dynamic_dropout=is_dynamic_dropout,
    )
    model.fit(
        x_train,
        y_train,
        n_epochs=100,
        validation_data=(x_test, y_test),
        batch_size=100,
        is_show=False,
    )

    score = model.model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    histories.append(model.history)

plt.figure(figsize=(12, 10))
legends_acc = sum(
    [
        [f'{index}_train acc', f'{index}_test acc']
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
plt.legend(legends_acc, loc='upper right')
plt.show()

plt.figure(figsize=(12, 10))
legends_loss = sum(
    [
        [f'{index}_train loss', f'{index}_test loss']
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

plt.legend(legends_loss, loc='upper right')
plt.show()
