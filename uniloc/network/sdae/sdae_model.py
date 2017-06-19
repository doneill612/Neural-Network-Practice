import tensorflow as tf
import sdae_graph
import sdae_train
import network.model

class SDAE(network.model.Model):
    '''
    Stacked Denoising Autoencoder model.
    '''    
    def __init__(self, name, config):
        '''
        params:
            `name`      : the model name - used for saving purposes
            `config`    : an SDAEConfig object containing hyperparameters and
                            TFRecord file paths.
        '''
        super(SDAE, self).__init__(name)
        self._config = config
    
    def build_computation_graph(self):
        return sdae_graph.build_graph(self._mode, self._config)
    
    def train(self):
        self._mode = 'train'
        graph = self.build_computation_graph()
        sdae_train.train(graph, self._session, self._config, self._name)
