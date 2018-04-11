import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

class LivePlot:

    def __init__(self, nn, epochs):
        self.nn = nn
        self.epochs = epochs
        self.fig = None
        self.ax = None
        self.ln = None

    def run(self):
        self.fig, self.ax = plt.subplots()
        self.ln, = plt.plot(self.nn.cost_history, 'bo', markersize=1, animated=True)
        ani = FuncAnimation(self.fig, self.update, frames=np.linspace(0, 10, 5), init_func=self.init, blit=True)
        plt.show()

    def init(self):
        self.nn.train_step()
        self.ax.set_xlim(0, self.epochs)
        self.ax.set_ylim(0, max(self.nn.cost_history))
        self.ax.set_ylabel('Cost')
        self.ax.set_xlabel('Epoch')
        return self.ln,

    def update(self, frames):
        self.nn.train_step()
        self.ln.set_data(range(0, len(self.nn.cost_history)), self.nn.cost_history)
        return self.ln,

class NeuralNetwork:

    # layers: list of numbers, each number is number of neurons in that hidden layer.
    #         Thus, length of list is number of hidden layers
    def __init__(self, n_inputs, n_outputs, h_layers, inputs, expected_outputs):
        self.n_inputs = n_inputs
        self.h_layers = h_layers
        self.n_outputs = n_outputs
        self.x = inputs
        self.y_ = expected_outputs
        self.input_layer = tf.placeholder(tf.float32, [None, self.n_inputs], name="InputLayer")
        self.expected_output_layer = tf.placeholder(tf.float32, [None, self.n_outputs],
                                                    name="ExpectedOutputLayer")
        self.weights = []
        self.bias = []
        self.cost_history = []
        self.learning_rate = 0.01
        self.output_layer = None
        self.session = None
        self.training_step = None
        self.cost_function = None

    def set_init_vals(self, is_weights_rand):
        if is_weights_rand:
            return tf.truncated_normal
        else:
            return tf.zeros

    def initialize_weights_bias(self, is_weights_rand=True):
        init_vals = self.set_init_vals(is_weights_rand)
        for i in range(0, len(self.h_layers) + 1):
            if i == 0:
                curr_weights = tf.Variable(init_vals([self.n_inputs, self.h_layers[0]]), name="Weights_Input_to_h1")
                curr_bias = tf.Variable(tf.truncated_normal([self.h_layers[0]]), name="Bias_h1")
            elif i < len(self.h_layers):
                curr_weights = tf.Variable(init_vals([self.h_layers[i - 1], self.h_layers[i]]),
                                           name='Weights_h{0}_to_h{1}'.format(i-1, i))
                curr_bias = tf.Variable(tf.truncated_normal([self.h_layers[i]]), name='Bias_h{0}'.format(i))
            else:
                curr_weights = tf.Variable(init_vals([self.h_layers[-1], self.n_outputs]),
                                           name="Weights_h{0}_to_output".format(i))
                curr_bias = tf.Variable(tf.truncated_normal([self.n_outputs]),
                                        name="Bias_outputLayer")

            self.weights.append(curr_weights)
            self.bias.append(curr_bias)

        init = tf.global_variables_initializer()
        self.session.run(init)

    def forward_propagation(self):
        curr_layer = tf.add(tf.matmul(self.input_layer, self.weights[0]), self.bias[0])
        curr_layer = tf.nn.sigmoid(curr_layer)

        for i in range(1, len(self.weights)):
            curr_layer = tf.add(tf.matmul(curr_layer, self.weights[i]), self.bias[i])
            curr_layer = tf.nn.sigmoid(curr_layer)

        self.output_layer = curr_layer

    def train(self, epochs=1000, learning_rate=0.01, isliveplot=False):
        self.learning_rate = learning_rate
        if isliveplot:
            self.live_plot(epochs)
        else:
            for i in range(epochs):
                self.train_step()

        # print('output: ', self.session.run(self.output_layer, feed_dict={self.input_layer: self.x}))
        # print('expected: ', self.y_)

    def train_step(self):
        curr_epoch = len(self.cost_history)
        self.session.run(self.training_step,
                         feed_dict={self.input_layer: self.x,
                                    self.expected_output_layer: self.y_})
        cost = self.session.run(self.cost_function, feed_dict={self.input_layer: self.x, self.expected_output_layer: self.y_})
        if curr_epoch % 10000 == 0:
            print('epoch: ', curr_epoch, ' cost: ', cost)
        self.cost_history.append(cost)

    def test(self, test_inputs, test_outputs):
        outputs = self.session.run(self.output_layer, feed_dict={self.input_layer: test_inputs})
        cost = self.session.run(self.cost_function, feed_dict={self.input_layer: test_inputs,
                                                               self.expected_output_layer: test_outputs})
        return outputs, cost

    def live_plot(self, epochs):
        lp = LivePlot(self, epochs)
        lp.run()

    def plot(self):
        plt.figure()
        plt.plot(self.cost_history)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.title('Learning Rate')
        plt.show()

    def open_session(self):
        self.session = tf.Session()

    def close_session(self):
        self.session.close()

    def make_model(self, is_weights_rand=True):
        self.initialize_weights_bias(is_weights_rand)
        self.forward_propagation()
        self.cost_function = tf.reduce_mean((self.output_layer - self.expected_output_layer) ** 2)
        self.training_step = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost_function)


    # To open tensor graph
    # cd to project dir then run
    # run python -m tensorboard.main --logdir="./"
    def make_tensor_board(self):
        tf.summary.FileWriter('./tensorgraph', self.session.graph)


'''

h_layers = [2]
inputs = [[0, 0],
          [1, 0],
          [0, 1],
          [1, 1]]

outputs = [[0],
           [1],
           [1],
           [0]]

nn = NeuralNetwork(n_inputs=2,
                   n_outputs=1,
                   h_layers=h_layers,
                   inputs=inputs,
                   expected_outputs=outputs)
nn.open_session()
nn.make_model(is_weights_rand=True)
nn.make_tensor_board()
s = time.time()
nn.train(epochs=99999, learning_rate=0.01, isliveplot=False)
e = time.time() - s
print("Training took ", e, "seconds")
nn.close_session()
nn.plot()

'''


