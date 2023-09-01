import numpy as np


class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, num_inputs, bias=1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias)."""
        self.weights = (np.random.rand(num_inputs + 1) * 2) - 1
        self.bias = bias

    def run(self, inputs):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(inputs, self.bias), self.weights)
        return self.sigmoid(x_sum)

    def set_weights(self, w_init):
        # w_init is a list of floats
        self.weights = np.array(w_init)

    def sigmoid(self, x_sum):
        # Return the output of the sigmoid function applied to x.
        return 1 / (1 + np.exp(-x_sum))


def main():
    perceptron = Perceptron(num_inputs=2)
    print("--Perceptron Initialization--")
    print(f'Weights: {perceptron.weights}')
    print(f'Bias: {perceptron.bias}')

    print("--AND-gate Perceptron Configuration--")
    perceptron.set_weights([10, 10, -15])
    print(f'Weights: {perceptron.weights}')
    print(f'Bias: {perceptron.bias}')

    print("--Testing AND-gate Perceptron--")
    print("0 0 = {0:.10f}".format(perceptron.run([0, 0])))
    print("0 1 = {0:.10f}".format(perceptron.run([0, 1])))
    print("1 0 = {0:.10f}".format(perceptron.run([1, 0])))
    print("1 1 = {0:.10f}".format(perceptron.run([1, 1])))

    print("--OR-gate Perceptron Configuration--")
    perceptron.set_weights([15, 15, -10])
    print(f'Weights: {perceptron.weights}')
    print(f'Bias: {perceptron.bias}')

    print("--Testing OR-gate Perceptron--")
    print("0 0 = {0:.10f}".format(perceptron.run([0, 0])))
    print("0 1 = {0:.10f}".format(perceptron.run([0, 1])))
    print("1 0 = {0:.10f}".format(perceptron.run([1, 0])))
    print("1 1 = {0:.10f}".format(perceptron.run([1, 1])))


if __name__ == '__main__':
    main()
