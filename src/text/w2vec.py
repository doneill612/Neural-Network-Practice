"""
Adapted from TensorFlow word2vec tutorial found online at
https://www.tensorflow.org/tutorials/word2vec
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
import tensorflow as tf
from six.moves import urllib

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

url = 'http://mattmahoney.net/dc/{}'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('training_steps', 100000, 'The total steps of training'
                                               'to perform. Default = 100000')
flags.DEFINE_integer('batch_size', 128, 'Batch size to be used in training.'
                                        'Default = 128')
flags.DEFINE_integer('embedding_size', 128, 'Dimension of word embedding vector.'
                                            'Default = 128')
flags.DEFINE_integer('skip_window', 1, 'How many words to consider around target.'
                                       'Default = 1')
flags.DEFINE_integer('num_skips', 2, 'How many times to reuse an input to generate.'
                                     'a label. Default = 2')
flags.DEFINE_integer('valid_size', 16, 'Size of the validation set.'
                                       'Default = 16')
flags.DEFINE_integer('valid_window', 100, 'Size of the validation window.'
                                          'Default = 100')
flags.DEFINE_integer('num_sampled', 64, 'Negative examples to sample.'
                                        'Default = 64')

def download_dataset(filename, expected_bytes):
    """Retrieve the dataset"""
    print('Fetching data from remote...')
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url.format(filename), filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('OK')
    else:
        msg = 'SIZE MISMATCH: size = {}'.format(statinfo.st_size)
        raise RuntimeError(msg)
    return filename

def read_dataset(filename):
    """Extract the dataset into memory"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def build_dataset(words, n_words):
    """Converts the raw data into a workable dataset"""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    mapping = dict()
    for k, v in count:
        mapping[k] = len(mapping)
    data = list()
    unk_count = 0
    for word in words:
        if word in mapping:
            index = mapping[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    inverse_mapping = dict(zip(mapping.values(), mapping.keys()))
    return data, count, mapping, inverse_mapping

data_index = 0

def generate_batch(data):
    """Obtain a training batch for skip-gram model"""
    global data_index
    batch = np.ndarray(shape=(FLAGS.batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(FLAGS.batch_size, 1), dtype=np.int32)
    span = 2 * FLAGS.skip_window + 1
    buf = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buf.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(FLAGS.batch_size // FLAGS.num_skips):
        target = FLAGS.skip_window
        targets_to_avoid = [FLAGS.skip_window]
        for j in range(FLAGS.num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * FLAGS.num_skips + j] = buf[FLAGS.skip_window]
            labels[i * FLAGS.num_skips + j, 0] = buf[target]
        if data_index == len(data):
            buf[:] = data[:span]
            data_index = span
        else:
            buf.append(data[data_index])
            data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

def build_graph(vocabulary_size):
    """Build the TensorFlow computation graph"""
    valid_examples = np.random.choice(FLAGS.valid_window,
                                      FLAGS.valid_size,
                                      replace=False)
    with tf.Graph().as_default() as graph:
        init_op = tf.global_variables_initializer()
        tf.add_to_collection('init_op', init_op)

        training_inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        training_labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, 1])
        tf.add_to_collection('training_inputs', training_inputs)
        tf.add_to_collection('training_labels', training_labels)
        validation_set = tf.constant(valid_examples, dtype=tf.int32)

        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, FLAGS.embedding_size], -1.0, 1.0))
        lookup = tf.nn.embedding_lookup(embeddings, training_inputs)
        weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, FLAGS.embedding_size],
                                    stddev=1.0 / math.sqrt(FLAGS.embedding_size)))
        biases = tf.Variable(tf.zeros([vocabulary_size]))

        cost = tf.reduce_mean(tf.nn.nce_loss(weights=weights,
                                             biases=biases,
                                             labels=training_labels,
                                             inputs=lookup,
                                             num_sampled=FLAGS.num_sampled,
                                             num_classes=vocabulary_size))
        tf.add_to_collection('cost', cost)
        optimizer = tf.train.GradientDescentOptimizer(1.0)
        with tf.name_scope('apply_grads'):
            train_op = optimizer.minimize(cost)
            tf.add_to_collection('train_op', train_op)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        validation_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                       validation_set)
        similarity = tf.matmul(validation_embeddings, normalized_embeddings,
                               transpose_b=True)
        tf.add_to_collection('similarity', similarity)
        return graph, normalized_embeddings, valid_examples

def train_model(vocabulary_size, inverse_mapping, data):
    """Trains a model given a computation graph"""
    graph, normalized_embeddings, valid_examples = (
            build_graph(vocabulary_size))
    with tf.Session(graph=graph) as sess:
        training_inputs = graph.get_collection('training_inputs')[0]
        training_labels = graph.get_collection('training_labels')[0]
        train_op = graph.get_collection('train_op')[0]
        cost = graph.get_collection('cost')[0]
        similarity = graph.get_collection('similarity')[0]
        average_loss = 0
        sess.run(tf.global_variables_initializer())
        for step in range(FLAGS.training_steps):
            batch_inputs, batch_labels = generate_batch(data)
            feed_dict = {training_inputs: batch_inputs,
                         training_labels: batch_labels}
            _, _cost = sess.run([train_op, cost], feed_dict=feed_dict)
            average_loss += _cost
            if step > 0 and step % 2000 == 0:
                average_loss /= 2000
                tf.logging.info('Average loss at step %d = %.3f',
                                step, average_loss)
                average_loss = 0
            if step > 0 and step % 10000 == 0:
                sim = similarity.eval()
                for i in range(FLAGS.valid_size):
                    valid_word = inverse_mapping[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to {}'.format(valid_word)
                    for k in range(top_k):
                        close_word = inverse_mapping[nearest[k]]
                        tf.logging.info('%s %s, ' % (log_str, close_word))
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()
        plot_results(final_embeddings, inverse_mapping)

def plot_results(final_embeddings, inverse_mapping, filename='tsne.png'):
    """Use Matplotlib to plot results"""
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [inverse_mapping[i] for i in range(plot_only)]
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

def main(argv=None):
    """Application entry point"""
    filename = download_dataset('text8.zip', 31344016)
    vocabulary = read_dataset(filename)
    vocabulary_size = 50000
    data, count, mapping, inverse_mapping = build_dataset(vocabulary,
                                                          vocabulary_size)
    del vocabulary
    print('Most common words (including UNK)', count[:5])
    print('Sample data', data[:10], [inverse_mapping[i] for i in data[:10]])
    train_model(vocabulary_size, inverse_mapping, data)

if __name__ == '__main__':
    tf.app.run(main=main)
