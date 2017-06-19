import tensorflow as tf
import sdae_graph
import sdae_config
import sdae_train

tf.logging.set_verbosity(tf.logging.INFO)

class SDAETrainTest(tf.test.TestCase):
    '''
    Unit testing. 
    Tests a training session on a TFRecord file.
    Unit test result:
    
    > & python ../uniloc/uniloc/network/sdae/sdae_train_test.py
    2017-06-19 16:21:22.977357: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library was
    n't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU co
    mputations.
    2017-06-19 16:21:22.977378: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library was
    n't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU co
    mputations.
    2017-06-19 16:21:22.977382: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library was
    n't compiled to use AVX instructions, but these are available on your machine and could speed up CPU compu
    tations.
    Global Step: 100 - Learning Rate: 0.001 - Loss: 0.0564454682171
    Global Step: 200 - Learning Rate: 0.001 - Loss: 0.0309845712036
    Global Step: 300 - Learning Rate: 0.001 - Loss: 0.0273861195892
    ...
    ...
    ...
    Global Step: 2800 - Learning Rate: 0.001 - Loss: 0.0153328701854
    Global Step: 2900 - Learning Rate: 0.001 - Loss: 0.0164202135056
    2017-06-19 16:23:48.333004: W tensorflow/core/framework/op_kernel.cc:1152] Out of range: RandomShuffleQueu
    e '_0_shuffle_batch/random_shuffle_queue' is closed and has insufficient elements (requested 100, current
    size 20)
            [[Node: shuffle_batch = QueueDequeueManyV2[component_types=[DT_FLOAT], timeout_ms=-1, _device="/j
    ob:localhost/replica:0/task:0/cpu:0"](shuffle_batch/random_shuffle_queue, shuffle_batch/n)]]
    ..
    ----------------------------------------------------------------------
    Ran 2 tests in 146.010s

    OK

    '''
    def setUp(self):
        self.config = sdae_config.default_configs.get('test_train_config')
        self.graph = sdae_graph.build_graph('train', self.config)
        self.session = None
    
    def testTrain(self):
        sdae_train.train(self.graph, self.session, self.config, 'test_train')
        

if __name__ == '__main__':
    tf.test.main()