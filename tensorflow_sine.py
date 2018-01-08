#!/usr/bin/env python

##
#
# Simple example of fitting a sine wave with
# a neural network using tensorflow
#
##

import numpy as np
import tensorflow as tf
import pylab as plt

N = 1000 # number of data points
t = np.linspace(0, 4*np.pi, N)
data = 3*np.sin(t+0.001) + 0.5 + np.random.randn(N)  # sine wave with additive noise

# Parameters
learning_rate = .1
training_epochs = 5000
batch_size = 200
display_step = 50

# Network Parameters
n_hidden_1 = 50 # 1st layer number of neurons
n_hidden_2 = 50 # 2nd layer number of neurons
n_input = 1 
n_classes = 1

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Dropout parameter
keep_prob = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.sigmoid(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)  # apply dropout to hidden layer
    # Hidden fully connected layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.sigmoid(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    # Output fully connected layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
pred = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.square(pred-Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(N/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # Get next batch
            batch_x = np.asarray(([[j] for j in t[batch_size*i:batch_size*(i+1)]]), dtype=np.float)  # put each entry as it's own row of the array
            batch_y = np.asarray(([[k] for k in data[batch_size*i:batch_size*(i+1)]]), dtype=np.float)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.9})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")


    # test the model!
    print("\nTesting the Model")
    long_t = np.linspace(-3*np.pi, 5*np.pi, N)   # test on data outside the range too
    test_x = np.asarray(([[j] for j in long_t]), dtype=np.float)
    # plot the original data
    #plt.plot(long_t,data, '.', color='0.75', label='original data')

    for i in range(100):
        # make and plot multiple predictions using dropout
        test_prediction = sess.run(pred, feed_dict={X: test_x, keep_prob: .8})
        test_prediction = np.ndarray.flatten(test_prediction)   # set as one long list for plotting
        plt.plot(long_t, test_prediction, 'b-', alpha=0.05)

    # and make one prediction without the dropout
    test_prediction = sess.run(pred, feed_dict={X: test_x, keep_prob: 1})  # don't use dropout in testing
    test_prediction = np.ndarray.flatten(test_prediction)   # set as one long list for plotting
    plt.plot(long_t, test_prediction, 'r-')

    plt.show()
