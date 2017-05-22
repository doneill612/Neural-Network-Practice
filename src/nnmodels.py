import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
"""
This file represents a more dynamic approach to creating
a fully-connected neural network. The fc_layer(...) function
allows for simple stacking of layers to create a custom neural
network model.

In this example, I decided to train a 4-layer deep neural network
on the MNIST hand-written digits data set. I am using Google's
renowned TensorFlow framework for the training of the network.

I hope to abstract out the functionality of the fc_layer and network_model
functions into classes to allow for even further facilitation of the creation
of the network.
"""
mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

# Placeholder for the flattened input images.
X = tf.placeholder("float", shape=[None, 784])
# Placeholder for the image labels.
y = tf.placeholder("float", shape=[None, 10])

# Represents a layer of the network.
# This is a rigid implementation, as the output is tailored to the specific
#   classification problem of recognizing the digits. We plan on using the
#   softmax_cross_entropy_with_logits function, so activation is not immediately
#   applied to the output. This is not the case in all neural networks.
def fc_layer(_input, channels_in, channels_out, output = False):
    weights = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[channels_out]))
    if output:
        act = tf.matmul(_input, weights) + biases
    else:
        act = tf.nn.relu(tf.add(tf.matmul(_input, weights), biases))
    return act

# Represents the neural network. I chose a 4-layer fully connected network model,
#   but more layers could have easily been added through additional calls of fc_layer.
def network_model():
    fc1 = fc_layer(X, 784, 256)
    fc2 = fc_layer(fc1, 256, 256)
    _logits = fc_layer(fc2, 256, 10, output=True)
    return _logits

# Forward propagation = logits
logits = network_model()

# Cost function.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=y))
# Using an AdamOptimizer in this implementation. Abstraction should allow
#   for the selection of an optimizer and learning rate.
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Initialization op for TensorFlow
init = tf.global_variables_initializer()

# Run the TensorFlow session.
with tf.Session() as sess:
    sess.run(init)
    # 15 epochs of training, batches of 100 images at a time.
    for i in range(15):
        batches = int(mnist.train.num_examples / 100)
        for batch in range(batches):
            batch_X, batch_Y = mnist.train.next_batch(100)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_X, y: batch_Y})
        print("Epoch ", i)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accracy: ", accuracy.eval({X: mnist.test.images, y: mnist.test.labels}))

"""The netowrk achieves 97.85% accuracy on average!"""