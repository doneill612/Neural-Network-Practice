from scipy import optimize
import numpy as np
import matplotlib.pyplot as plot

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalization of input parameters
X = X / np.amax(X, axis=0)
y = y / 100


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidPrime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


# A rudimentary 3 layer Neural Network capable of predicting test scores given hours of sleep and hours
#       studied for the exam. Project inspired by Stephen C Welch (@stephencwelch Twitter) who has a great
#       repository on github with a walkthrough on building this network. My implementation is a restructured
#       version of what he provides, with organizational improvements.

# noinspection PyShadowingNames,PyShadowingNames,PyShadowingNames,
# PyShadowingNames,PyShadowingNames,PyShadowingNames,PyShadowingNames
class Network(object):
    def __init__(self):
        # Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        self.reg = 0.0001

        # Weights (initially randomized)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

        # Unactivated layer 2 neuron values
        self.z2 = None
        # Layer 2 activity (activated layer 2 neuron values)
        self.a2 = None
        # Unactivated layer 3 neuron values
        self.z3 = None
        # Layer 3 activity (activated layer 3 neuron values - the neural network output)
        self.yHat = None

    # Calculates the error-squared between supervised learning values and network output after
    #   one forward propagation
    def costFunction(self, X, y):
        self.yHat = self.forwardPropagate(X)
        # noinspection PyTypeChecker
        J = 0.5 * sum((y - self.yHat) ** 2) / X.shape[0] + \
            (self.reg / 2) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return J

    # Propagates an input vector through the neural network
    def forwardPropagate(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = sigmoid(self.z2)
        # noinspection PyTypeChecker
        self.z3 = np.dot(self.a2, self.W2)
        yHat = sigmoid(self.z3)
        return yHat

    # Forward-propagates the input vector and calculates weight gradients at each layer of the network
    def backPropogate(self, X, y):
        self.yHat = self.forwardPropagate(X)

        delta3 = np.multiply(-(y - self.yHat), sigmoidPrime(self.z3))
        # Gradient of the L2-L3 synapses
        dJdW2 = np.dot(self.a2.T, delta3) / X.shape[0] + self.reg * self.W2

        delta2 = np.dot(delta3, self.W2.T) * sigmoidPrime(self.z2)
        # Gradient of the L1-L2 synapses
        dJdW1 = np.dot(X.T, delta2) / X.shape[0] + self.reg * self.W1

        return dJdW1, dJdW2

    # Retrieves the synapse weights at each layer of the network and concatenates them into a single vector
    # Used for efficiently passing weight information to other data structures
    def getParams(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    # Used for updating the weights of the network from outside the network structure
    def setParams(self, params):
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    # Returns the gradients of each layer after a forward propagation in a single vector
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.backPropogate(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

# An evaluator is used for validating numerical gradient calculations.
# Ensures the calculus is being performed correctly during gradient calculation in the network by
#       comparing the calculated gradients with the limit definition of a derivative
#
#     f'(x) = lim(eps->0) { [f(x + eps) - f(x)] / eps } -- This calculation should yield results
#                                                          similar to the gradient calculation method
#
class Evaluator(object):
    def __init__(self, network):
        self.network = network
        self.numericalGradient = None
    def computeNumericalGradients(self):
        params = self.network.getParams()
        numericalGradient = np.zeros(params.shape)
        perturbation = np.zeros(params.shape)
        eps = 1e-4

        for p in range(len(params)):
            perturbation[p] = eps
            self.network.setParams(params + perturbation)
            lossPos = self.network.costFunction()
            self.network.setParams(params - perturbation)
            lossNeg = self.network.costFunction()

            numericalGradient[p] = (lossPos - lossNeg) / (2 * eps)
            perturbation[p] = 0

        self.network.setParams(params)
        self.numericalGradient = numericalGradient
    def evaluate(self):
        gradients = self.network.computeGradients()
        print(self.numericalGradient)
        print(gradients)

# Trainer is used to train a neural network based on input data via (in this implementation)
#   supervised learning.
# Optimization is performed via Broyden-Fletcher-Goldfarb-Shanno numerical optimization
#      (https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)

# noinspection PyShadowingNames,PyShadowingNames,PyShadowingNames,PyShadowingNames
class Trainer(object):
    def __init__(self, network):
        self.network = network

        self.X = None
        self.y = None
        self.J = None
        self.optimizationResults = None

    def callback(self, params):
        self.network.setParams(params)
        self.J.append(self.network.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.network.setParams(params)
        cost = self.network.costFunction(X, y)
        grad = self.network.computeGradients(X, y)
        return cost, grad

    def train(self, X, y):
        self.X = X
        self.y = y
        self.J = []

        params0 = self.network.getParams()

        options = {'maxiter': 5000, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',
                                 args=(X, y), options=options, callback=self.callback)
        self.network.setParams(_res.x)
        self.optimizationResults = _res


nn = Network()
t = Trainer(nn)
t.train(X, y)
y_hat = nn.forwardPropagate(X)
print("Neural Network Results:")
for i in range(len(y)):
    print("Actual Test Score: ", y[i] * 100)
    print("Neural Network prediction: ", y_hat[i] * 100)
print("See training results in plot.")
plot.xlabel("Iterations")
plot.ylabel("Cost")
plot.grid(1)
plot.plot(t.J)
plot.show()
