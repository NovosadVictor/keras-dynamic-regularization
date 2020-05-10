from constants.utils import \
    load_diabetes_dataset
from constants.classes.model import NNModel

model = NNModel()
x_train, y_train = load_diabetes_dataset()


model.fit(x_train, y_train, n_epochs=100)

_, accuracy = model.model.evaluate(x_train, y_train)
print(accuracy)
