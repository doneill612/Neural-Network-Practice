import tensorflow as tf
import os

class SDAEConfig(object):

    '''
    A Stacked Denoising Autoencoder network configuration object. The configuration
    object is used to pass network hyperparameters and a set of TFRecord file paths
    to the SDAE model.
    '''
    def __init__(self, batch_size, 
                    epochs, learning_rate,
                    activation, layer_sizes, tf_record_file_paths):
        '''
        params:
            `batch_size`            : the batch size to be used when training the network
            `epochs`                : the number of training epochs to run during a training session
            `learning_rate`         : the learning rate passed to the training optimizer
            `activation`            : the activation function to use in the hidden layers of the network
            `layer_sizes`           : a list of layer sizes, each index corresponding to a layer
            `tf_record_file_paths`  : a list of file paths pointing to TFRecords
        '''
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._activation = activation
        self._layer_sizes = layer_sizes
        self._tf_record_file_paths = tf_record_file_paths
        
    
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def epochs(self):
        return self._epochs
    @property
    def activation(self):
        return self._activation
    @property
    def layer_sizes(self):
        return self._layer_sizes
    @property
    def learning_rate(self):
        return self._learning_rate
    @property
    def tf_record_file_paths(self):
        return self._tf_record_file_paths


default_configs = {
    'test_train_config': SDAEConfig(batch_size=100, 
        epochs=15, activation='sigmoid', learning_rate=0.001,
        layer_sizes=[784, 512, 512, 256], tf_record_file_paths=os.path.join('{}/data_handling/tfrecords'
                        .format(os.path.dirname(os.path.realpath(__file__))), 
                        '{}_tfrecords.tfrecords'.format('mnist_train.csv')))
}