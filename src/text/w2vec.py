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

def __inverse_of(mapping):
    return dict(zip(mapping.values(), mapping.keys()))

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
    inverse_mapping = __inverse_of(mapping)
    return data, count, mapping, inverse_mapping

data_index = 0

def generate_batch(data, batch_size, num_skips, skip_window):
    """Obtain a training batch for skip-gram model"""
    global data_index
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buf = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buf.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buf[skip_window]
            labels[i * num_skips + j, 0] = buf[target]
        if data_index == len(data):
            buf[:] = data[:span]
            data_index = span
        else:
            buf.append(data[data_index])
            data_index += 1
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

def build_graph(vocabulary_size, embedding_size):
    """Build the TensorFlow computation graph"""
    valid_examples = np.random.choice(FLAGS.valid_window,
                                      FLAGS.valid_size,
                                      replace=False)
    with tf.Graph().as_default() as graph:
        init_op = tf.global_variables_initializer()
        tf.add_to_collection('init_op', init_op)

        training_inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        training_labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, 1])
        validation_set = tf.constant(valid_examples, dtype=tf.int32)

        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        lookup = tf.nn.embedding_lookup(embeddings, training_inputs)
        weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
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
        return graph

def train_model(graph):
    """Trains a model given a computation graph"""
    with tf.Session(graph=graph):
        # TODO implement
        return None

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

'''
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)
'''
