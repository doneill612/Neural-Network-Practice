from scipy import optimize
import numpy as np

examples = np.array(([1, 0], [0, 0], [1, 1], [1, 1], [1, 0], [0, 1], [1, 1]), dtype=int)
testSet = np.array(([0, 0], [1, 0], [0, 1], [1, 1], [1, 0], [0, 1], [0, 0]), dtype=int)
testAnswers = np.array(([0], [1], [1], [0], [1], [1], [0]), dtype=int)
results = np.array(([1], [0], [0], [0], [1], [1], [0]), dtype=int)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidPrime(z):
    #    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)
    return sigmoid(z) * (1 - sigmoid(z))

class Network(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayersize = 1

        self.X = examples
        self.Y = results

        self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.w2 = np.random.randn(self.hiddenLayerSize, self.outputLayersize)

        self.reg = 0.0001

        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.yHat = None

    def feedForward(self):
        self.z2 = np.dot(self.X, self.w1)
        self.a2 = sigmoid(self.z2)
        # noinspection PyTypeChecker
        self.z3 = np.dot(self.a2, self.w2)
        self.yHat = sigmoid(self.z3)

    def forwardPropagate(self, example):
        self.z2 = np.dot(example, self.w1)
        self.a2 = sigmoid(self.z2)
        # noinspection PyTypeChecker
        self.z3 = np.dot(self.a2, self.w2)
        self.yHat = sigmoid(self.z3)

    def cost(self):
        self.feedForward()
        J = 0.5 * sum((self.Y - self.yHat) ** 2) / self.X.shape[0] + \
            (self.reg / 2) * (np.sum(self.w1 ** 2) + np.sum(self.w2 ** 2))
        return J

    def backProp(self):
        self.feedForward()

        delta3 = np.multiply(-(self.Y - self.yHat), sigmoidPrime(self.z3))
        dJdW2 = (np.dot(self.a2.T, delta3) / self.X.shape[0]) + self.reg * self.w2

        delta2 = np.dot(delta3, self.w2.T) * sigmoidPrime(self.z2)
        dJdW1 = (np.dot(self.X.T, delta2) / self.X.shape[0]) + self.reg * self.w1

        return dJdW1, dJdW2

    def computeGradients(self):
        dJdW1, dJdW2 = self.backProp()
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def getParams(self):
        return np.concatenate((self.w1.ravel(), self.w2.ravel()))

    def setParams(self, params):
        w1_end = self.hiddenLayerSize * self.inputLayerSize
        self.w1 = np.reshape(params[0 : w1_end], (self.inputLayerSize, self.hiddenLayerSize))
        w2_end = w1_end + (self.hiddenLayerSize * self.outputLayersize)
        self.w2 = np.reshape(params[w1_end : w2_end], (self.hiddenLayerSize, self.outputLayersize))

class Trainer(object):
    def __init__(self, network):
        self.network = network

        self.J = None
        self.optimizedResults = None

    def callback(self, params):
        self.network.setParams(params)
        self.J.append(self.network.cost())

    def wrapper(self, params, X, y):
        self.network.setParams(params)
        cost = self.network.cost()
        grad = self.network.computeGradients()
        return cost, grad

    def train(self):
        self.J = []

        initialParams = self.network.getParams()
        options = {'maxiter' : 5000, 'disp' : True}
        opResults = optimize.minimize(self.wrapper, initialParams, jac = True, method = 'BFGS',
                                        args = (self.network.X, self.network.Y),
                                        options = options,
                                        callback = self.callback)
        self.network.setParams(opResults.x)
        self.optimizedResults = opResults

nn = Network()
nn.forwardPropagate(testSet)
for i in range(len(nn.yHat)):
    print("Expected: ", testAnswers[i][0])
    print("Returned (UNTRAINED): ", 1 if nn.yHat[i] > 0.5 else 0)
t = Trainer(nn)
t.train()
nn.forwardPropagate(testSet)
for i in range(len(nn.yHat)):
    print("Expected: ", testAnswers[i][0])
    print("Returned: ", 1 if nn.yHat[i] > 0.5 else 0)
