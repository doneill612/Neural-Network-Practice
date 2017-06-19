import tensorflow as tf
import abc

class Model(object):
    '''
    Abstract base class for neural network models in TensorFlow.
    '''
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        '''
        params:
            `name`      : the model name - used for saving purposes
            `mode`      : the computation graph building mode - either 'train' or 'use'
            `session`   : the TensorFlow session to run the computation graph
        '''
        self._session = None
        self._mode = None
        self._name = name
    
    @abc.abstractmethod
    def build_computation_graph(self):
        '''
        Constructs the TensorFlow computation graph.
        Must be implemented in subclasses.

        returns:
            a `tf.Graph()` object
        '''
        pass
    
    @abc.abstractmethod
    def train(self):
        '''
        Trains a network model. Must be implemented in subclasses.
        '''
        pass
    
    def load_model(self, checkpoint_file, metagraph_file):
        '''
        Restores a checkpoint and metagraph to the model session.

        args:
            `checkpoint_file`   : the location of the checkpoint file to be loaded
            `metagraph_file`    : the location of the metagraph information to be loaded
        '''
        with tf.Graph().as_default():
            self._session = tf.Session()
            saver = tf.train.import_meta_graph(metagraph_file)
            saver.restore(self._session, checkpoint_file)
            tf.logging.info('Checkpoint restored from {}.'
                        .format(checkpoint_file))
    
    def save_model(self, checkpoint_file):
        '''
        Saves a model to a checkpoint file with metagraph data.

        args:
            `checkpoint_file`: the location of the checkpoint file to be saved
        '''
        with self._session.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._session, checkpoint_file, write_meta_graph=True)
            tf.logging.info('Checkpoint saved to {} with MetaGraph.'
                        .format(checkpoint_file))

        
