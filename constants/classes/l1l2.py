from keras.regularizers import Regularizer
from keras import backend as K


class DynamicL1L2(Regularizer):
    def __init__(self, l1=0.0, l2=0.01, is_dynamic=True):
        with K.name_scope(self.__class__.__name__):
            self.is_dynamic = is_dynamic
            if self.is_dynamic:
                self.l1 = K.variable(l1, name='l1')
                self.l2 = K.variable(l2, name='l2')
            else:
                self.l1 = l1
                self.l2 = l2

            self.val_l1 = l1
            self.val_l2 = l2

    def set_l1_l2(self, l1, l2):
        K.set_value(self.l1, l1)
        K.set_value(self.l2, l2)
        self.val_l1 = l1
        self.val_l2 = l2

    def __call__(self, x):
        regularization = 0.
        if self.val_l1 > 0.:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.val_l2 > 0.:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        if self.is_dynamic:
            config = {
                'l1': float(K.get_value(self.l1)),
                'l2': float(K.get_value(self.l2)),
            }
        else:
            config = {
                'l1': self.l1,
                'l2': self.l2,
            }

        return config
