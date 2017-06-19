import tensorflow as tf
import math
import data_handling.record_utils as record_utils

def _assertions(mode, config):
    '''
    Asserts valid SDAE configuration and valid graph construction mode.

    args:
        `mode`      : the graph construction mode for the SDAE
        `config`    : the SDAE configuration
    raises:
        `ValueError` if:
            - mode is not 'train' or 'use'
            - activation function specified by config is not 'sigmoid',
                'relu', or 'tanh'
    '''
    if mode not in ('train', 'test', 'use'):
        tf.logging.fatal('Bad config passed to graph builder.')
        raise ValueError("The config mode parameter \
            must be 'train', 'test', or 'use'.")
        
    if config.activation not in ('sigmoid', 'relu', 'tanh'):
        tf.logging.fatal('Bad config passed to graph builder.')
        raise ValueError("The config activation parameter \
            must be 'sigmoid', 'relu', or 'tanh'.")

def _corrupt(input_tensor, p_corruption):
    '''
    Corrupts an input tensor element-wise. Similar to the dropout principle, this method
    randomly sets elements in `input_tensor` to zero.

    args:
        `input_tensor`  : the tensor to corrupt
        `p_corruption`  : the probability that the input tensor will be corrupted. In training,
                            this value is 1.0, and in validation and use mode this value is 0.0
    returns:
        `corrupted_inputs`: the corrupted tensor 
    '''
    if p_corruption == 1:
        corrupted_inputs = tf.multiply(input_tensor, tf.cast(tf.random_uniform(
                                                shape=tf.shape(input_tensor),
                                                minval=0,
                                                maxval=2,
                                                dtype=tf.int32), tf.float32))
    else:
        corrupted_inputs = input_tensor
    
    return corrupted_inputs

def _read_tf_records(tf_record_file_paths, batch_size, epochs, input_layer_size):
    '''
    Builds the placeholders for the TensorFlow graph in 'train' mode. In training, examples (inputs) are
    read from TFRecord Example protos. See `/data_handling/record_utils.py` documentation for a more concrete description.
    More information on TFRecord objects and Example protos can be found at https://www.tensorflow.org/programmers_guide/reading_data.
    A placeholder is built for the corruption probability. Both the inputs and the placeholder for the 
    corruption probability are placed in the graph collection so they can be referenced during training.

    args:
        `tf_record_file_paths`: a list of file paths to TFRecords containing Example protos.
        `batch_size`: the size of the batch of training examples to parse from the Example protos.
        `input_layer_size`: the size of the input data
    returns:
        `inputs`        : a placeholder containing a batch of inputs
        `p_corruption`  : the corruption probability placeholder
    '''
    if not isinstance(tf_record_file_paths, list):
        tf_record_file_paths = [tf_record_file_paths]
    
    inputs = record_utils.get_record_batch(tf_record_file_paths, batch_size, epochs, input_layer_size)
    inputs = tf.identity(inputs, name='inputs')

    p_corruption = 1.0

    tf.add_to_collection('inputs', inputs)
    

    return inputs, p_corruption

def _build_placeholders(input_layer_size):
    '''
    Builds the placeholders for the TensorFlow graph in 'use' mode. The placeholders are also placed in the
    graph collection so they can be referenced and fed during training.
    
    args:
        `input_layer_size`: the size of the input data used for allocating the placeholder shape
    returns:
        `inputs`        : the inputs placeholder
        `p_corruption`  : the corruption probability placeholder
    '''
    inputs = tf.placeholder(tf.float32, [None, input_layer_size], name='inputs')
    
    p_corruption = 0.0
    tf.add_to_collection('inputs', inputs)
    

    return inputs, p_corruption

def _build_encoder_ops(layer_sizes, corrupted_inputs, encoder_weights, activate):
    '''
    Builds the encoder portion of the SDAE. The weights and biases of the encoder
    are named so they can be later loaded into a Classifier model.
    
    args:
        `layer_sizes`       : the SDAE layer sizes
        `corrupted_inputs`  : the corrupted inputs
        `encoder_weights`   : a list to store the weight variables so they can be used to
                                construct the decoder portion of the SDAE
        `activate`          : the activation function
    returns:
        `encoder` : the encoder output (latent representation of the SDAE)
    '''
    for index, next_output_size in enumerate(layer_sizes[1:]):
        next_input_size = int(corrupted_inputs.get_shape()[1])
        weights = tf.Variable(tf.truncated_normal(shape=[next_input_size, next_output_size],
                                stddev=0.25),
                                name='encoder_weights_l{}'.format(str(index)))
        biases = tf.Variable(tf.constant(0.1, shape=[next_output_size]), 
                                name='encoder_biases_l{}'.format(str(index)))
        encoder_weights.append(weights)
        corrupted_inputs = activate(tf.add(tf.matmul(corrupted_inputs, weights), biases))
    
    encoder = tf.identity(corrupted_inputs)
    return encoder

def _build_decoder_ops(layer_sizes, encoder, encoder_weights, activate):
    '''
    Builds the decoder portion of the SDAE. The decoder weights are intially equivalent
    to the encoder weights.
    
    args:
        `layer_sizes`       : the SDAE layer sizes
        `corrupted_inputs`  : the corrupted inputs
        `encoder_weights`   : a list consisting of the encoder weights
        `activate`          : the activation function
    returns:
        `decoder` : the decoder output (output SDAE vector)
    '''
    for index, next_output_size in enumerate(layer_sizes[:-1][::-1]):
        weights = tf.transpose(encoder_weights[index])
        biases = tf.Variable(tf.constant(0.1, shape=[next_output_size]))
        encoder = activate(tf.add(tf.matmul(encoder, weights), biases))

    decoder = tf.identity(encoder)
    return decoder

def _build_training_ops(outputs, inputs, learning_rate):
    '''
    Builds the graph training ops. A global_step variable is created
    for keeping track of training progress. The cost, global step,
    and training op are all added to the graph collection.
    
    args:
        `outputs`       : the decoder outputs
        `inputs`        : the original input data
        `learning_rate` : the learning rate specified by the SDAE config
    '''
    global_step = tf.Variable(0, trainable=False, name='global_step')
    tf.add_to_collection('global_step', global_step)

    #cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=inputs)
    #cost = tf.sqrt(tf.reduce_mean(tf.square(outputs - inputs)))
    cost = tf.reduce_mean(tf.squared_difference(outputs, inputs))
    tf.add_to_collection('cost', cost)
    
    

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    tf.add_to_collection('train_op', train_op)
    
def build_graph(mode, config):
    '''
    Builds the TensorFlow computation graph for an SDAE.

    args:
        `mode`      : the graph construction mode for the SDAE
        `config`    : the SDAE configuration
    returns:
        `graph`: the computation graph
    '''
    _assertions(mode, config)

    layer_sizes = config.layer_sizes
    tf_record_file_paths = config.tf_record_file_paths
    
    if config.activation == 'sigmoid':
        activate = tf.nn.sigmoid
    elif config.activation == 'relu':
        activate = tf.nn.relu
    elif config.activation == 'tanh':
        activate = tf.nn.tanh

    with tf.Graph().as_default() as graph:

        if mode == 'train':
            inputs, p_corruption = _read_tf_records(tf_record_file_paths, config.batch_size, config.epochs, layer_sizes[0])
        else:
            inputs, p_corruption = _build_placeholders(layer_sizes[0])
        
        corrupted_inputs = _corrupt(inputs, p_corruption)
        
        encoder_weights = list()
        encoder = _build_encoder_ops(layer_sizes, corrupted_inputs, encoder_weights, activate) 
        
        # Only weights up to this layer will be loaded when linking the SDAE to a classifier.
        latent_representation = tf.identity(encoder, name='latent_representation')
          
        encoder_weights.reverse()
        decoder = _build_decoder_ops(layer_sizes, encoder, encoder_weights, activate)
                
        outputs = tf.identity(decoder, name='outputs')

        if mode == 'train': 
            _build_training_ops(outputs, inputs, config.learning_rate)
        
        return graph