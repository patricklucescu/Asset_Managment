import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt


# Constants
X_DIM = 1 # 2
INPUT_SIZE    = X_DIM
RNN_HIDDEN    = 2
OUTPUT_SIZE   = 1
# TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 5e-4
LEARNING_RATE_DECAY = 0.99
USE_LSTM = True
map_fn = tf.map_fn
#
#
# def as_bytes(num, final_size):
#     res = []
#     for _ in range(final_size):
#         res.append(num % 2)
#         num //= 2
#     return res


def generate_sine_batch(batch_size):

    sine_size = 1000
    sine_period = 20.0
    t = np.array(range(sine_size))

    source_noise_amplitude = 0.
    observation_noise_amplitude = 0.

    s = np.sin(2*np.pi*t/sine_period) + np.random.randn(*(t.shape)) * source_noise_amplitude

    num_features = 1

    assert(X_DIM == 2)
    x = np.zeros((num_features, batch_size, X_DIM))

    x[0, 0: batch_size, 0] = s[0: batch_size]
    x[0, 0: batch_size-1, 1] = s[1: batch_size]

    y = np.zeros((num_features, batch_size, 1))
    y[0, 0:-2, 0] = x[0, 2:, 0]

    y = y + np.random.randn(*(y.shape)) * observation_noise_amplitude

    return x, y


def generate_sine_seq_batch(batch_size, seq_length=2):

    sine_size = 1000
    assert(batch_size+seq_length < sine_size)

    sine_period = 20.0
    t = np.array(range(sine_size))

    source_noise_amplitude = 0.
    observation_noise_amplitude = 0.

    s = np.sin(2*np.pi*t/sine_period) + np.random.randn(*(t.shape)) * source_noise_amplitude
    num_features = seq_length
    x = np.zeros((num_features, batch_size, 1))

    for i in range(seq_length):
        x[i, :, 0] = s[i: batch_size+i]

    y = np.zeros((1, batch_size, 1))
    y[0, :, 0] = s[2: batch_size+2]
    y = y + np.random.randn(*y.shape) * observation_noise_amplitude

    return x, y


class Model:
    def __init__(self, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, use_lstm=True):
        # graph input/output size
        self.input_size = input_size
        self.output_size = output_size

        # session
        self.session = None

        # graph nodes
        self.use_lstm = use_lstm
        self.inputs = None
        self.outputs = None
        self.predicted_outputs = None
        self.error = None
        self.train_fn = None
        # learning parameter
        self.learning_rate = None

    def build(self):

        tf.reset_default_graph()
        cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
        self.learning_rate = tf.placeholder(tf.float32, shape=())  # (time, batch, in)
        self.inputs = tf.placeholder(tf.float32, (None, None, self.input_size))  # (time, batch, in)
        self.outputs = tf.placeholder(tf.float32, (None, None, self.output_size)) # (time, batch, out)

        batch_size = tf.shape(self.inputs)[1]
        initial_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=initial_state, time_major=True)

        def final_projection(x):
            return layers.linear(x, num_outputs=self.output_size, activation_fn=None)

        # apply projection to every timestep.
        self.predicted_outputs = map_fn(final_projection, rnn_outputs)
        error = tf.squared_difference(self.predicted_outputs[-1, :, :], self.outputs)
        self.error = tf.reduce_mean(error)

        # optimize
        self.train_fn = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(error)
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def train_batch(self, current_learning_rate=LEARNING_RATE, batch_size=40, iterations=500, generate_batch=generate_sine_batch):

        epoch_error = 0

        for _ in range(iterations):
            x, y = generate_batch(batch_size)
            epoch_error += self.session.run([self.error, self.train_fn], {
                self.inputs: x,
                self.outputs: y,
                self.learning_rate: current_learning_rate
            })[0]

        return epoch_error

    def predict_batch(self, valid_x):
        valid_prediction = self.session.run([self.predicted_outputs], {
            self.inputs: valid_x,
        })
        return valid_prediction

    def validate_batch(self, valid_x, valid_y):
        valid_mse, valid_prediction = self.session.run([self.error, self.predicted_outputs], {
            self.inputs: valid_x,
            self.outputs: valid_y,
        })
        return valid_mse, valid_prediction


def test_train_validate_seq():

    generate_batch = generate_sine_seq_batch
    # validation
    valid_x, valid_y = generate_batch(batch_size=100)

    model = Model(INPUT_SIZE, OUTPUT_SIZE, USE_LSTM)
    model.build()
    current_learning_rate = LEARNING_RATE

    for epoch in range(1000):
        epoch_error = model.train_batch(current_learning_rate, generate_batch=generate_batch)
        validation_mse, valid_prediction = model.validate_batch(valid_x, valid_y)
        current_learning_rate = current_learning_rate * LEARNING_RATE_DECAY
        print("Epoch %d, train error: %.2f, valid error: %.1f" % (epoch, epoch_error, validation_mse))
        plt.plot(valid_prediction[1, :, 0]); plt.plot(valid_y[0, :, 0]); plt.show()


if __name__ == '__main__':
    test_train_validate_seq()
