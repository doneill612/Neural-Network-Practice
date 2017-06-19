import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', os.path.join(os.path.dirname(
                                        os.path.realpath(__file__)),
                                        'data_handling/trained_models'),
                           'The directory where trained models'
                           'are to be stored.')
tf.app.flags.DEFINE_integer('update_freq', 100,
                            'The number of training steps to take '
                            'before logging a training update message.')
tf.app.flags.DEFINE_integer('save_freq', 25, 
                            'The time interval in seconds for which to '
                            'saved a model checkpoint file.')
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            'The maxmimum number of global steps to take '
                            'during model training.')

def train(graph, sess, config, name):
    '''
    Trains an SDAE network model. Training is performed with the help of
    a TensorFlow `Supervisor` object which oversees the training process.
    The supervisor will save the model periodically as specified by the 
    corresponding app flag. TF App flags are defined prior to training is
    executed and their docstrings are declared in the DEFINE function 
    (see above).

    args:
        `graph`: the `Graph` containing the training ops
        `sess`: the SDAE session property (casted in a `with` block)
        `config`: the SDAE network configuration
        `name`: the model name used for saving.
    '''
  

    inputs = graph.get_collection('inputs')[0]
    global_step = graph.get_collection('global_step')[0]
    train_op = graph.get_collection('train_op')[0]
    cost = graph.get_collection('cost')[0]

    learning_rate = config.learning_rate

    log_dir = os.path.join(FLAGS.train_dir, name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    supervisor = tf.train.Supervisor(graph=graph, logdir=log_dir,
                                     save_model_secs=FLAGS.save_freq, 
                                     global_step=global_step,
                                     checkpoint_basename='{}.ckpt'.format(name))
    
    with supervisor.managed_session() as sess:
        tf.logging.info('Beginning training...')
        _global_step = sess.run(global_step)
        if _global_step >= FLAGS.max_steps:
            tf.logging.info('This model has already exceeded the maximum allocated '
                                'training steps. Aborting training.')
            return
        while _global_step < FLAGS.max_steps:
            if supervisor.should_stop():
                break
            if (_global_step + 1) % FLAGS.update_freq == 0:
                (_global_step, _cost, _,) = sess.run([global_step, cost, train_op])
                tf.logging.info('Global Step: %d - '
                                'Learning Rate: %.5f - '
                                'Loss: %.3f - ',
                                _global_step, learning_rate, _cost)
                                
            else:
                (_global_step, _) = sess.run([global_step, train_op])
        supervisor.saver.save(sess, supervisor.save_path, global_step=_global_step)
        tf.logging.info('Training complete.')