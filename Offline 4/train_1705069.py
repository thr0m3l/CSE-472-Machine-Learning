import numpy as np
import pickle


# Data preprocessing

import numpy as np
import os
import glob
import cv2
import pandas as pd
import pickle
import matplotlib.pyplot as plt


data_dir = os.path.join('./NumtaDB_with_aug', '')
paths_train_a = sorted(
    glob.glob(os.path.join(data_dir, 'training-a', '*.png')))
paths_train_b = sorted(
    glob.glob(os.path.join(data_dir, 'training-b', '*.png')))
paths_train_e = sorted(
    glob.glob(os.path.join(data_dir, 'training-e', '*.png')))
paths_train_c = sorted(
    glob.glob(os.path.join(data_dir, 'training-c', '*.png')))
paths_train_d = sorted(
    glob.glob(os.path.join(data_dir, 'training-d', '*.png')))
paths_train_all = paths_train_a + paths_train_b + paths_train_c


path_label_train_a = os.path.join(data_dir, 'training-a.csv')
path_label_train_b = os.path.join(data_dir, 'training-b.csv')
path_label_train_e = os.path.join(data_dir, 'training-e.csv')
path_label_train_c = os.path.join(data_dir, 'training-c.csv')
path_label_train_d = os.path.join(data_dir, 'training-d.csv')


def get_key(path):
    # seperates the key of an image from the filepath
    key = path.split(sep=os.sep)[-1]
    return key


def f1_score(y_true, y_pred, average='macro'):
    # Calculate the F1 score
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average=average)


def contrast_grey(img):
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    return (high-low)/(high+low), high, low


def adjust_contrast_grey(img, target=0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./(high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0), np.minimum(
            np.full(img.shape, 255), img)).astype(np.uint8)
    return


def one_hot(x, num_classes=10):

    # Convert x to numpy array
    x = np.array(x)

    out = np.zeros((x.shape[0], num_classes))

    # Convert each label to a one hot vector
    out = np.eye(num_classes)[x]

    return out


def get_data(paths_img, path_label=None, resize_dim=None):
    '''reads images from the filepaths, resizes them (if given), and returns them in a numpy array
    Args:
        paths_img: image filepaths
        path_label: pass image label filepaths while processing training data, defaults to None while processing testing data
        resize_dim: if given, the image is resized to resize_dim x resize_dim (optional)
    Returns:
        X: group of images
        y: categorical true labels
    '''
    X = []
    for i, path in enumerate(paths_img):
        img = cv2.imread(path)  # images loaded in color (BGR)

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if resize_dim is not None:
            try:
                # resize image to 28x28
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            except:
                print('Error while resizing')

        # Convert the image to binary
        img = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Check if the background is black or white
        # If the background is white, then invert the image
        if np.mean(img) > 128:
            img = 255 - img

        # Dilate and erode to remove noise
        # Kernel for dilation
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        row_drop = []

        for i in range(img.shape[0]):
            if 0 <= np.mean(img[i, :]) <= 5:
                row_drop.append(i)

        col_drop = []
        for i in range(img.shape[1]):
            if 0 <= np.mean(img[:, i]) <= 5:
                col_drop.append(i)

        # Drop the rows and columns
        img = np.delete(img, row_drop, axis=0)
        img = np.delete(img, col_drop, axis=1)

        # Resize the image back to 28x28
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        X.append(img)  # expand image to 28x28x1 and append to the list

        if i == len(paths_img)-1:
            end = '\n'
        else:
            end = '\r'
        print('processed {}/{}'.format(i+1, len(paths_img)), end=end)

    X = np.array(X)  # tranform list to numpy array
    if path_label is None:
        return X
    else:
        df = pd.read_csv(path_label)  # read labels
        df = df.set_index('filename')
        # Sort the dataframe by the column named 'filename'
        df = df.sort_values(by=['filename'])
        # Get the labels from the dataframe corresponding to the image names
        # get the labels corresponding to the images
        y_label = [df.loc[get_key(path)]['digit'] for path in paths_img]
        # transfrom integer value to categorical variable
        y = one_hot(y_label, 10)
        return X, y


def load():
    X_train_a, y_train_a = get_data(
        paths_train_a, path_label_train_a, resize_dim=28)
    X_train_b, y_train_b = get_data(
        paths_train_b, path_label_train_b, resize_dim=28)
    X_train_c, y_train_c = get_data(
        paths_train_c, path_label_train_c, resize_dim=28)
    # X_train_d, y_train_d=get_data(paths_train_d,path_label_train_d,resize_dim=28)
    # X_train_e, y_train_e=get_data(paths_train_e,path_label_train_e,resize_dim=28)

    X_train_a = X_train_a.reshape(X_train_a.shape[0], 28, 28, 1)
    X_train_b = X_train_b.reshape(X_train_b.shape[0], 28, 28, 1)
    X_train_c = X_train_c.reshape(X_train_c.shape[0], 28, 28, 1)
    # X_train_d = X_train_d.reshape(X_train_d.shape[0], 28, 28, 1)
    # X_train_e = X_train_e.reshape(X_train_e.shape[0], 28, 28, 1)

    X_train_all = np.concatenate((X_train_a, X_train_b, X_train_c), axis=0)
    y_train_all = np.concatenate((y_train_a, y_train_b, y_train_c), axis=0)

    # Normalize the data
    X_train_all = X_train_all.astype('float32') / 255

    indices = list(range(len(X_train_all)))
    print(len(indices))
    # np.random.seed(42)
    np.random.shuffle(indices)

    ind = int(len(indices)*0.80)
    # train data
    X_train = X_train_all[indices[:ind]]
    y_train = y_train_all[indices[:ind]]
    # validation data
    X_val = X_train_all[indices[-(len(indices)-ind):]]
    y_val = y_train_all[indices[-(len(indices)-ind):]]

    # Convert from one hot encoding to integer
    y_val = np.argmax(y_val, axis=1)

    print('X_train shape:', X_train.shape)
    print('X_val shape:', X_val.shape)

    return X_train, y_train, X_val, y_val


# Activation functions

class ReLU():
    def f(self, x):
        return np.maximum(0, x)

    def df(self, x, cached_y=None):
        return np.where(x <= 0, 0, 1)


class SoftMax():
    def f(self, x):
        y = np.exp(x - np.max(x, axis=1, keepdims=True))
        return y / np.sum(y, axis=1, keepdims=True)

    def df(self, x, cached_y=None):
        return np.where(x <= 0, 0, 1)


relu = ReLU()
softmax = SoftMax()

# Cost functions

epsilon = 1e-20


class SoftmaxCrossEntropy():
    def f(self, a_last, y):
        batch_size = y.shape[0]
        cost = -1 / batch_size * \
            (y * np.log(np.clip(a_last, epsilon, 1.0))).sum()
        return cost

    def grad(self, a_last, y):
        # To avoid vanishing gradients
        return - np.divide(y, np.clip(a_last, epsilon, 1.0))


softmax_cross_entropy = SoftmaxCrossEntropy()

# Optimizer functions


class GradientDescent():
    def __init__(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def initialize(self):
        pass

    def update(self, learning_rate, w_grads, b_grads, step):
        for layer in self.trainable_layers:
            layer.update_params(dw=learning_rate * w_grads[layer],
                                db=learning_rate * b_grads[layer])


gradient_descent = GradientDescent


class Adam():
    def __init__(self, trainable_layers, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.trainable_layers = trainable_layers
        self.v = {}
        self.s = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def initialize(self):
        for layer in self.trainable_layers:
            w, b = layer.get_params()
            w_shape = w.shape
            b_shape = b.shape
            self.v[('dw', layer)] = np.zeros(w_shape)
            self.v[('db', layer)] = np.zeros(b_shape)
            self.s[('dw', layer)] = np.zeros(w_shape)
            self.s[('db', layer)] = np.zeros(b_shape)

    def update(self, learning_rate, w_grads, b_grads, step):
        v_correction_term = 1 - np.power(self.beta1, step)
        s_correction_term = 1 - np.power(self.beta2, step)
        s_corrected = {}
        v_corrected = {}

        for layer in self.trainable_layers:
            layer_dw = ('dw', layer)
            layer_db = ('db', layer)

            self.v[layer_dw] = (self.beta1 * self.v[layer_dw] +
                                (1 - self.beta1) * w_grads[layer])
            self.v[layer_db] = (self.beta1 * self.v[layer_db] +
                                (1 - self.beta1) * b_grads[layer])

            v_corrected[layer_dw] = self.v[layer_dw] / v_correction_term
            v_corrected[layer_db] = self.v[layer_db] / v_correction_term

            self.s[layer_dw] = (self.beta2 * self.s[layer_dw] +
                                (1 - self.beta2) * np.square(w_grads[layer]))
            self.s[layer_db] = (self.beta2 * self.s[layer_db] +
                                (1 - self.beta2) * np.square(b_grads[layer]))

            s_corrected[layer_dw] = self.s[layer_dw] / s_correction_term
            s_corrected[layer_db] = self.s[layer_db] / s_correction_term

            dw = (learning_rate * v_corrected[layer_dw] /
                  (np.sqrt(s_corrected[layer_dw]) + self.epsilon))
            db = (learning_rate * v_corrected[layer_db] /
                  (np.sqrt(s_corrected[layer_db]) + self.epsilon))

            layer.update_params(dw, db)


adam = Adam

######## LAYERS######

# Convolutional Layer


class Conv():
    """2D convolutional layer.

    Attributes
    ----------
    kernel_size : int
        Height and Width of the 2D convolution window.
    stride : int
        Stride along height and width of the input volume on which the convolution is applied.
    padding: str
        Padding mode, 'valid' or 'same'.
    pad : int
        Padding size.
    n_h : int
        Height of the output volume.
    n_w : int
        Width of the output volume.
    n_c : int
        Number of channels of the output volume. Corresponds to the number of filters.
    n_h_prev : int
        Height of the input volume.
    n_w_prev : int
        Width of the input volume.
    n_c_prev : int
        Number of channels of the input volume.
    w : numpy.ndarray
        Weights.
    b : numpy.ndarray
        Biases.
    activation : Activation
        Activation function applied to the output volume after performing the convolution operation.
    cache : dict
        Cache.
    """

    def __init__(self, kernel_size, stride, n_c, padding='valid', activation=relu):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pad = None
        self.n_h, self.n_w, self.n_c = None, None, n_c
        self.n_h_prev, self.n_w_prev, self.n_c_prev = None, None, None
        self.w = None
        self.b = None
        self.activation = activation
        self.cache = {}

    def init(self, in_dim):
        self.pad = 0 if self.padding == 'valid' else int(
            (self.kernel_size - 1) / 2)

        self.n_h_prev, self.n_w_prev, self.n_c_prev = in_dim
        self.n_h = int((self.n_h_prev - self.kernel_size +
                       2 * self.pad) / self.stride + 1)
        self.n_w = int((self.n_w_prev - self.kernel_size +
                       2 * self.pad) / self.stride + 1)

        # Xavier initialization
        self.w = np.random.randn(self.kernel_size, self.kernel_size, self.n_c_prev, self.n_c) * \
            np.sqrt(2 / (self.kernel_size * self.kernel_size * self.n_c_prev))
        self.b = np.zeros((1, 1, 1, self.n_c))

    def forward(self, a_prev, training):
        batch_size = a_prev.shape[0]
        a_prev_padded = Conv.zero_pad(a_prev, self.pad)
        out = np.zeros((batch_size, self.n_h, self.n_w, self.n_c))

        # Convolve
        for i in range(self.n_h):
            v_start = i * self.stride
            v_end = v_start + self.kernel_size

            for j in range(self.n_w):
                h_start = j * self.stride
                h_end = h_start + self.kernel_size

                out[:, i, j, :] = np.sum(a_prev_padded[:, v_start:v_end, h_start:h_end, :, np.newaxis] *
                                         self.w[np.newaxis, :, :, :], axis=(1, 2, 3))

        z = out + self.b
        a = self.activation.f(z)

        if training:
            # Cache for backward pass
            self.cache.update({'a_prev': a_prev, 'z': z, 'a': a})

        return a

    def backward(self, da):
        batch_size = da.shape[0]
        a_prev, z, a = (self.cache[key] for key in ('a_prev', 'z', 'a'))
        a_prev_pad = Conv.zero_pad(
            a_prev, self.pad) if self.pad != 0 else a_prev

        da_prev = np.zeros(
            (batch_size, self.n_h_prev, self.n_w_prev, self.n_c_prev))
        da_prev_pad = Conv.zero_pad(
            da_prev, self.pad) if self.pad != 0 else da_prev

        dz = da * self.activation.df(z, cached_y=a)
        db = 1 / batch_size * dz.sum(axis=(0, 1, 2))
        dw = np.zeros((self.kernel_size, self.kernel_size,
                      self.n_c_prev, self.n_c))

        # 'Convolve' back
        for i in range(self.n_h):
            v_start = self.stride * i
            v_end = v_start + self.kernel_size

            for j in range(self.n_w):
                h_start = self.stride * j
                h_end = h_start + self.kernel_size

                da_prev_pad[:, v_start:v_end, h_start:h_end, :] += \
                    np.sum(self.w[np.newaxis, :, :, :, :] *
                           dz[:, i:i+1, j:j+1, np.newaxis, :], axis=4)

                dw += np.sum(a_prev_pad[:, v_start:v_end, h_start:h_end, :, np.newaxis] *
                             dz[:, i:i+1, j:j+1, np.newaxis, :], axis=0)

        dw /= batch_size

        return da_prev, dw, db

    def get_output_dim(self):
        return self.n_h, self.n_w, self.n_c

    def update_params(self, dw, db):
        self.w -= dw
        self.b -= db

    def get_params(self):
        return self.w, self.b

    def clear_cache(self):
        self.cache = {}

    @staticmethod
    def zero_pad(x, pad):
        return np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')


# Max Pooling Layer

class Pool():

    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.n_h, self.n_w, self.n_c = None, None, None
        self.n_h_prev, self.n_w_prev, self.n_c_prev = None, None, None
        self.w = None
        self.b = None
        self.cache = {}

    def init(self, in_dim):
        self.n_h_prev, self.n_w_prev, self.n_c_prev = in_dim
        self.n_h = int((self.n_h_prev - self.pool_size) / self.stride + 1)
        self.n_w = int((self.n_w_prev - self.pool_size) / self.stride + 1)
        self.n_c = self.n_c_prev

    def forward(self, a_prev, training):
        batch_size = a_prev.shape[0]
        a = np.zeros((batch_size, self.n_h, self.n_w, self.n_c))

        # Pool
        for i in range(self.n_h):
            v_start = i * self.stride
            v_end = v_start + self.pool_size

            for j in range(self.n_w):
                h_start = j * self.stride
                h_end = h_start + self.pool_size

                a_prev_slice = a_prev[:, v_start:v_end, h_start:h_end, :]

                if training:
                    # Cache for backward pass
                    self.cache_max_mask(a_prev_slice, (i, j))

                a[:, i, j, :] = np.max(a_prev_slice, axis=(1, 2))

        if training:
            self.cache['a_prev'] = a_prev

        return a

    def backward(self, da):
        a_prev = self.cache['a_prev']
        batch_size = a_prev.shape[0]
        da_prev = np.zeros(
            (batch_size, self.n_h_prev, self.n_w_prev, self.n_c_prev))

        # 'Pool' back
        for i in range(self.n_h):
            v_start = i * self.stride
            v_end = v_start + self.pool_size

            for j in range(self.n_w):
                h_start = j * self.stride
                h_end = h_start + self.pool_size

                # Max pooling
                da_prev[:, v_start:v_end, h_start:h_end, :] += da[:,
                                                                  i:i+1, j:j+1, :] * self.cache[(i, j)]

        return da_prev, None, None

    def cache_max_mask(self, x, ij):
        mask = np.zeros_like(x)

        # This would be like doing idx = np.argmax(x, axis=(1,2)) if that was possible
        reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        idx = np.argmax(reshaped_x, axis=1)

        ax1, ax2 = np.indices((x.shape[0], x.shape[3]))
        mask.reshape(mask.shape[0], mask.shape[1] *
                     mask.shape[2], mask.shape[3])[ax1, idx, ax2] = 1
        self.cache[ij] = mask

    def update_params(self, dw, db):
        pass

    def get_params(self):
        pass

    def clear_cache(self):
        self.cache = {}

    def get_output_dim(self):
        return self.n_h, self.n_w, self.n_c

# Flatten Layer


class Flatten():
    """Flatten layer.

    Attributes
    ----------
    original_dim : tuple
        Shape of the input ndarray.
    output_dim : tuple
        Shape of the output ndarray.
    """

    def __init__(self):
        super().__init__()
        self.original_dim = None
        self.output_dim = None

    def init(self, in_dim):
        from functools import reduce
        self.original_dim = in_dim
        self.output_dim = reduce(lambda x, y: x * y, self.original_dim)

    def forward(self, a_prev, training):
        return a_prev.reshape(a_prev.shape[0], -1)

    def backward(self, da):
        return da.reshape(da.shape[0], *self.original_dim), None, None

    def get_params(self):
        pass

    def update_params(self, dw, db):
        pass

    def clear_cache(self):
        pass

    def get_output_dim(self):
        return self.output_dim

# Fully Connected Layer


class FullyConnected():
    """Densely connected layer.

    Attributes
    ----------
    size : int
        Number of neurons.
    activation : Activation
        Neurons' activation's function.
    is_softmax : bool
        Whether or not the activation is softmax.
    cache : dict
        Cache.
    w : numpy.ndarray
        Weights.
    b : numpy.ndarray
        Biases.
    """

    def __init__(self, size, activation):
        self.size = size
        self.activation = activation
        self.is_softmax = isinstance(self.activation, SoftMax)
        self.cache = {}
        self.w = None
        self.b = None

    def init(self, in_dim):
        # He initialization
        self.w = np.random.randn(self.size, in_dim) * np.sqrt(2 / in_dim)

        self.b = np.zeros((1, self.size))

    def forward(self, a_prev, training):
        z = np.dot(a_prev, self.w.T) + self.b
        a = self.activation.f(z)

        if training:
            # Cache for backward pass
            self.cache.update({'a_prev': a_prev, 'z': z, 'a': a})

        return a

    def backward(self, da):
        a_prev, z, a = (self.cache[key] for key in ('a_prev', 'z', 'a'))
        batch_size = a_prev.shape[0]

        if self.is_softmax:
            # Get back y from the gradient wrt the cost of this layer's activations
            # That is get back y from - y/a = da
            y = da * (-a)

            dz = a - y
        else:
            dz = da * self.activation.df(z, cached_y=a)

        dw = 1 / batch_size * np.dot(dz.T, a_prev)
        db = 1 / batch_size * dz.sum(axis=0, keepdims=True)
        da_prev = np.dot(dz, self.w)

        return da_prev, dw, db

    def update_params(self, dw, db):
        self.w -= dw
        self.b -= db

    def get_params(self):
        return self.w, self.b

    def clear_cache(self):
        self.cache = {}

    def get_output_dim(self):
        return self.size

# Neural Network class


class NeuralNetwork:
    """Neural network model.

    Attributes
    ----------
    layers : list
        Layers used in the model.
    w_grads : dict
        Weights' gradients during backpropagation.
    b_grads : dict
        Biases' gradients during backpropagation.
    cost_function : CostFunction
        Cost function to be minimized.
    optimizer : Optimizer
        Optimizer used to update trainable parameters (weights and biases).
    l2_lambda : float
        L2 regularization parameter.
    trainable_layers: list
        Trainable layers(those that have trainable parameters) used in the model.
    """

    def __init__(self, input_dim, layers, cost_function, optimizer=gradient_descent, l2_lambda=0):
        self.layers = layers
        self.w_grads = {}
        self.b_grads = {}
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda

        # Store training loss, validation loss and accuracy and macro f1 score to plot them later
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []
        self.val_f1 = []

        # Initialize the layers in the model providing the input dimension they should expect
        self.layers[0].init(input_dim)
        for prev_layer, curr_layer in zip(self.layers, self.layers[1:]):
            curr_layer.init(prev_layer.get_output_dim())

        self.trainable_layers = set(
            layer for layer in self.layers if layer.get_params() is not None)
        self.optimizer = optimizer(self.trainable_layers)
        self.optimizer.initialize()

    def forward_prop(self, x, training=True):

        a = x
        for layer in self.layers:
            a = layer.forward(a, training)

        return a

    def backward_prop(self, a_last, y):

        da = self.cost_function.grad(a_last, y)
        batch_size = da.shape[0]

        for layer in reversed(self.layers):
            da_prev, dw, db = layer.backward(da)

            if layer in self.trainable_layers:
                if self.l2_lambda != 0:
                    # Update the weights' gradients also wrt the l2 regularization cost
                    self.w_grads[layer] = dw + \
                        (self.l2_lambda / batch_size) * layer.get_params()[0]
                else:
                    self.w_grads[layer] = dw

                self.b_grads[layer] = db

            da = da_prev

    def predict(self, x):

        a_last = self.forward_prop(x, training=False)
        return a_last

    def update_param(self, learning_rate, step):

        self.optimizer.update(learning_rate, self.w_grads, self.b_grads, step)

    def compute_cost(self, a_last, y):
        from functools import reduce
        cost = self.cost_function.f(a_last, y)
        if self.l2_lambda != 0:
            batch_size = y.shape[0]
            weights = [layer.get_params()[0]
                       for layer in self.trainable_layers]
            l2_cost = (self.l2_lambda / (2 * batch_size)) * \
                reduce(lambda ws, w: ws + np.sum(np.square(w)), weights, 0)
            return cost + l2_cost
        else:
            return cost

    def train(self, x_train, y_train, mini_batch_size, learning_rate, num_epochs, validation_data):

        x_val, y_val = validation_data
        self.learning_rate = learning_rate

        # Convert y_val to one hot encoding for calculating loss
        y_val_ = np.eye(10)[y_val.reshape(-1)]

        print(
            f"Started training [batch_size={mini_batch_size}, learning_rate={learning_rate}]")
        step = 0
        for e in range(num_epochs):
            print("Epoch " + str(e + 1))
            epoch_cost = 0

            if mini_batch_size == x_train.shape[0]:
                mini_batches = (x_train, y_train)
            else:
                mini_batches = NeuralNetwork.create_mini_batches(
                    x_train, y_train, mini_batch_size)

            num_mini_batches = len(mini_batches)
            for i, mini_batch in enumerate(mini_batches, 1):
                mini_batch_x, mini_batch_y = mini_batch
                step += 1
                epoch_cost += self.train_step(mini_batch_x, mini_batch_y,
                                              learning_rate, step) / mini_batch_size
                print("\rProgress {:1.1%}".format(
                    i / num_mini_batches), end="")

            print(f"\nCost after epoch {e+1}: {epoch_cost}")

            print("Computing accuracy on validation set...")
            accuracy = np.sum(np.argmax(self.predict(x_val),
                              axis=1) == y_val) / x_val.shape[0]
            print(f"Accuracy on validation set: {accuracy}")
            f1 = f1_score(y_val, np.argmax(
                self.predict(x_val), axis=1), average='macro')
            print(f"F1 score on validation set: {f1}")
            val_loss = self.compute_cost(self.predict(x_val), y_val_)
            print(f"Loss on validation set: {val_loss}")
            training_loss = self.compute_cost(self.predict(x_train), y_train)
            print(f"Loss on training set: {training_loss}")

            # Append the loss and accuracy to the lists to plot them later
            self.train_loss.append(training_loss)
            self.val_loss.append(val_loss)
            self.val_acc.append(accuracy)
            self.val_f1.append(f1)

        print("Finished training")

        # Clear all the cached values
        self.clear_cache()

    def train_step(self, x_train, y_train, learning_rate, step):

        a_last = self.forward_prop(x_train, training=True)
        self.backward_prop(a_last, y_train)
        cost = self.compute_cost(a_last, y_train)
        self.update_param(learning_rate, step)
        return cost

    def clear_cache(self):
        # Clear the cached values of all the layers

        for layer in self.layers:
            layer.clear_cache()

    def plot(self):
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # Plot the loss
        ax1.plot(self.train_loss, label='Training loss')
        ax2.plot(self.val_loss, label='Validation loss')
        ax1.set_ylabel('Training Loss')
        ax2.set_ylabel('Validation Loss')
        ax1.set_xlabel('Epoch')
        # Set the x axis values to numbers from 0 to the number of epochs
        ax1.set_xticks(np.arange(0, len(self.train_loss), 1))
        ax2.set_xticks(np.arange(0, len(self.val_loss), 1))
        ax2.set_xlabel('Epoch')

        # Add plot title
        plt.title('Learning rate = ' + str(self.learning_rate))

        # Save the figure
        plt.savefig('loss_5.png')

        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # Plot the accuracy and f1 score
        ax1.plot(self.val_acc, label='Validation accuracy')
        ax2.plot(self.val_f1, label='Validation f1 score')
        ax1.set_xlabel('Epoch')
        ax2.set_xlabel('Epoch')
        ax1.set_ylabel('Validation Accuracy')
        ax2.set_ylabel('Validation F1 Score')

        ax1.set_xticks(np.arange(1, len(self.val_acc), 1))
        ax2.set_xticks(np.arange(1, len(self.val_f1), 1))

        # Add plot title
        plt.title('Learning rate = ' + str(self.learning_rate))
        # save the figure
        plt.savefig('acc_5.png')

    @staticmethod
    def create_mini_batches(x, y, mini_batch_size):

        batch_size = x.shape[0]
        mini_batches = []

        p = np.random.permutation(x.shape[0])
        x, y = x[p, :], y[p, :]
        num_complete_minibatches = batch_size // mini_batch_size

        for k in range(0, num_complete_minibatches):
            mini_batches.append((
                x[k * mini_batch_size:(k + 1) * mini_batch_size, :],
                y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
            ))

        # Fill with remaining data, if needed
        if batch_size % mini_batch_size != 0:
            mini_batches.append((
                x[num_complete_minibatches * mini_batch_size:, :],
                y[num_complete_minibatches * mini_batch_size:, :]
            ))

        return mini_batches


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load()

    cnn = NeuralNetwork(
        input_dim=(28, 28, 1),
        layers=[
            Conv(5, 1, 6, activation=relu),
            Pool(2, 2),
            Conv(5, 1, 16, activation=relu),
            Pool(2, 2),
            Flatten(),
            FullyConnected(120, relu),
            FullyConnected(84, relu),
            FullyConnected(10, softmax),
        ],
        optimizer=gradient_descent,
        cost_function=softmax_cross_entropy
    )

    cnn.train(x_train, y_train,
              mini_batch_size=200,
              learning_rate=0.005,
              num_epochs=30,
              validation_data=(x_test, y_test))

    cnn.plot()

    # save the model to disk as a pickle file
    pickle.dump(cnn, open("cnn_model_1.pkl", "wb"))
