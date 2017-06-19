import tensorflow as tf
import sdae_graph
import sdae_config

tf.logging.set_verbosity(tf.logging.INFO)

class SDAEGraphTest(tf.test.TestCase):
    '''
    Unit testing. 
    Tests both graph building modes.
    Unit test result:
    
    > & python ../uniloc/uniloc/network/sdae/sdae_graph_test.py
    ...
    -----------------------------------------------------------
    Ran 3 tests in 0.336s

    OK

    '''
    def setUp(self):
        self.config = sdae_config.default_configs.get('test_train_config')
    
    def testBuildForTraining(self):
        graph = sdae_graph.build_graph('train', self.config)
        
        self.assertTrue(isinstance(graph, tf.Graph))
        self.assertIn('inputs', graph.get_all_collection_keys())
        self.assertIn('cost', graph.get_all_collection_keys())
        self.assertIn('global_step', graph.get_all_collection_keys())
        self.assertIn('train_op', graph.get_all_collection_keys())
    
    def testBuildForUse(self):
        graph = sdae_graph.build_graph('use', self.config)
        
        self.assertTrue(isinstance(graph, tf.Graph))
        self.assertIn('inputs', graph.get_all_collection_keys())
        self.assertNotIn('cost', graph.get_all_collection_keys())
        self.assertNotIn('global_step', graph.get_all_collection_keys())
        self.assertNotIn('train_op', graph.get_all_collection_keys())

if __name__ == '__main__':
    tf.test.main()