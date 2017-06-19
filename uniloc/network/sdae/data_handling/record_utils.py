import csv
import os
import tensorflow as tf


def csv_to_tf_record(csv_file_name):
    '''
    Reads a .csv file containing training data and converts each row
    into Example protos. Example protos are protocol buffers 
    (https://www.tensorflow.org/programmers_guide/reading_data#file_formats 
    see 'Standard TensorFlow format' section) which contain trainable feature information.
    The Example protos are serialized into String format, and then saved to a .tfrecords file.
    These files are in a binary format and constitute a `TFRecord` object. They can be accessed with 
    a `TFRecordReader` obect which deserializes the Example protos and feeds the data to a tensor.

    args:
        `csv_file_name`: the file name of the .csv file to be converted to a .tfrecords file (csv is 
                         assumed to be at the location `/tfrecords/csv_file_name.csv`)
    '''
    full_path = os.path.join('{}/tfrecords'.format(os.path.dirname(os.path.realpath(__file__))), 
                            csv_file_name)
    f = open(full_path, 'r')
    try:
        reader = csv.reader(f)
        writer_path = os.path.join('{}/tfrecords'.format(os.path.dirname(
                                    os.path.realpath(__file__))), 
                                        '{}_tfrecords.tfrecords'.format(csv_file_name))
        with tf.python_io.TFRecordWriter(writer_path) as writer:
            for row in reader:
                if csv_file_name == 'mnist_train.csv':
                    row = row[1:]
                row_float = list(map(float, row))
                max_val = max(row_float)
                row_float = [row_float[i] / max_val for i in range(len(row_float))]
                feature = dict(input_vec=tf.train.Feature(float_list=
                                            tf.train.FloatList(value=row_float)))
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
    finally:
        f.close()
    
            
def get_record_batch(file_list, batch_size, epochs, input_size, threads=3):
    '''
    Reads and deserializes `batch_size` Example protos from a specified list of files.
    The batch is shuffled, and when created, spawns `QueueRunner` objects.
    Each of these objects hold a list of enqueue ops for a queue to run in a thread.

    args:
        `file_list`     : a list of file paths to .tfrecords files containing Example protos
        `batch_size`    : the size of the batch to pull
        `input_size`    : the input layer size of the network (essentially a validation
                          parameter which ensures that the Example proto features are of 
                          the appropriate length)
        `threads`       : the number of threads to be used when running enqueue ops
    returns:
        `inputs`: a batch of inputs to be used in training
    '''
    input_producer = tf.train.string_input_producer(file_list, num_epochs=epochs)
    reader = tf.TFRecordReader()
    serialized_example = reader.read(input_producer)[1]
    
    features = dict(input_vec=tf.FixedLenFeature(shape=[input_size], dtype=tf.float32))

    parsed = tf.parse_single_example(serialized_example, features=features)
    parsed_as_float = tf.cast(parsed['input_vec'], tf.float32)

    inputs = tf.train.shuffle_batch([parsed_as_float], batch_size=batch_size, 
                    capacity=500, min_after_dequeue=10, num_threads=threads)
    
    return inputs
