import record_utils
import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)

class TFRecordTests(tf.test.TestCase):
    '''
    Unit testing. 
    Tests both reading and writing of TFRecords.
    Unit test result:
    
    > & python c:/Users/David/Documents/David/work/uniloc/uniloc/network/sdae/data_handling/record_utils_test.py
    Batch size = 10, output shape = (10, 3)
    ...
    -----------------------------------------------------------
    Ran 3 tests in 0.512s

    OK
    '''
    def setUp(self):
        file_path = os.path.join('{}\\tfrecords'.format(os.path.dirname(os.path.realpath(__file__))), 
                        '{}_tfrecords.tfrecords'.format('test_csv_tfrecords'))
        self.file_list = [file_path]
    
    def testTFRecordWriter(self):
        record_utils.csv_to_tf_record('test_csv.csv')

    def testTFRecordReader(self):
        batch = record_utils.get_record_batch(self.file_list, 10, 15, 3)
        tf.logging.info('Batch size = 10, output shape = {}'.format(batch.get_shape()))



if __name__ == '__main__':
    tf.test.main()