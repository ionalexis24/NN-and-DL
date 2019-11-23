#### Libraries
# Standard library
import pickle
import gzip
import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return training_data, validation_data, test_data


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data)


training_data, validation_data, test_data = load_data_wrapper()


# td = list(training_data.__next__()[0].reshape(28, 28))  # gen training_data e zip/iterabil. Daca vrei alt elem plm
# plt.imshow(td, cmap=plt.get_cmap('gray'))  # asa afisezi imaginea lol
# plt.show()

# training_data, validation_data, test_data = load_data()
# X_train, y_train = training_data[0], training_data[1]  # asa afisezi din raw data...
# # print(X_train.shape, y_train.shape)
# plt.imshow(np.array(X_train[0]).reshape(28, 28), cmap=plt.get_cmap('gray'))
# plt.show()

class Network(object):
    def __init__(self, sizes):
        """
        Initializam reteaua cu arhitectura pasata la size biases si weights random
        :param sizes: lista sa moara jean gen [input,hidden...,output], hidden poate sa fie si vid
        """
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Gen net(a)=predictie. More like net.feedforward(a)...
        :param a: vector cu dim cat input.
        :return: vector cu dim cat output.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        """
        Facem update la param retelei cu stocastic grad descent.
        :param training_data: cica intra o lista de tuplete (x,y), da eu am vazut ca
                            e o lista de liste de doua np.array-uri...
        :param epochs: de cate ori se trece prin training set
        :param mini_batch_size: cat de mare e mini-setul de date
        :param eta: gen $\eta$
        :param test_data: None sau ceva, in caz de ceva, dupa fiecare epoca,
                        reteaua vede cat de bine a invatat pe acest set de date lol.
                        ia mai mult dar merita
        :return:  Nu returnam drq nimic, doar updatam parametrii retelei
        """

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):                             # pentru fiecare epoca
            random.shuffle(training_data)                   # amesteca setul de date
            mini_batches = [
                training_data[k:k+mini_batch_size]          # imparte setul in miniset-uri de mini_batch_size
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:                 # ia un miniset
                self.update_mini_batch(mini_batch, eta)     # faci update la parametri cu el

            if test_data:                                   # daca avem test data, evaluam
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:                                           # daca nu, zicem buna ziua frumos
                print("Epoch {} complete".format(j))        # si respectuos dupa fiecare epoca

    def update_mini_batch(self, mini_batch, eta):
        """
        Actualizam parametrii retelei
        :param mini_batch: minisetul de date pasat in metoda de mai sus
        :param eta: gen $\eta$
        :return: nimic, doar actualizeaza param.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]   # init $\nabla b$ si $\nabla w$
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # gen la dimeniuni cu b si w
        for x, y in mini_batch:                                                 #pentru fiecare exemplu din miniset
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)                  #calc deriv in toate b^l_j si toate w^l_{j,k}
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]     #pentru fiecare j,k,l facem suma deriv
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]     #
        self.weights = [w - (eta / len(mini_batch)) * nw                        # (20)
                        for w, nw in zip(self.weights, nabla_w)]                #pentru fiecare j,k,l updatam param
        self.biases = [b - (eta / len(mini_batch)) * nb                         # (21)
                       for b, nb in zip(self.biases, nabla_b)]                  #

#### Miscellaneous functions

def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
