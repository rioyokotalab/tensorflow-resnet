from resnet import * 
import tensorflow as tf
from tensorflow.python.client import timeline

try:
    import cPickle as pickle
except:
    import pickle

import datetime
import os

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('log_dir', '/opt/storage/logs/resnet/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 16, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 100, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'wheather to log device placement')
tf.app.flags.DEFINE_float('per_process_gpu_memory_fraction', 0.3, "gpu option")

def get_log_dir(dataset_name):
    t = datetime.datetime.today()
    dt_str = t.strftime("%Y%m%d%H%M%S")
    filename = dataset_name + "_" +  dt_str
    return os.path.join(FLAGS.log_dir, filename)

def get_tensors_for_cifar(g):
    i = [
           g.get_tensor_by_name("cond/Merge:0"),
           g.get_tensor_by_name("scale1/Conv2D:0"), 
           g.get_tensor_by_name("scale1/block1/Relu:0"), 
           g.get_tensor_by_name("scale1/block2/Relu:0"), 
           g.get_tensor_by_name("scale1/block3/Relu:0"), 
           g.get_tensor_by_name("scale2/block1/Relu:0"), 
           g.get_tensor_by_name("scale2/block2/Relu:0"), 
           g.get_tensor_by_name("scale2/block3/Relu:0"), 
           g.get_tensor_by_name("scale3/block1/Relu:0"), 
           g.get_tensor_by_name("scale3/block2/Relu:0"), 
           g.get_tensor_by_name("scale3/block3/Relu:0"),
           g.get_tensor_by_name("avg_pool:0"), 
           g.get_tensor_by_name("fc/xw_plus_b:0")
        ]
    return i

def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size

def train(is_training, logits, images, labels, dataset_name):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    loss_ = loss(logits, labels)
    predictions = tf.nn.softmax(logits)

    top1_error = top_k_error(predictions, labels, 1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)

    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)

    tf.summary.scalar('learning_rate', FLAGS.learning_rate)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.image_summary('images', images)

        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    # for profile
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #run_metadata = tf.RunMetadata()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
   
    log_dir = get_log_dir(dataset_name)
    print "Training logs will be stored in", log_dir

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(log_dir)
        if not latest:
            print "No checkpoint to continue from in", log_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)

    for x in xrange(FLAGS.max_steps + 1):
        start_time = time.time()

        step = sess.run(global_step)
        i = [train_op, loss_]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)

        o = sess.run(i, { is_training: True })

        #for profile
        #o = sess.run(i, { is_training: True }, options=run_options, run_metadata=run_metadata)
        #step_stats = run_metadata.step_stats
        #tl = timeline.Timeline(step_stats)
        #ctf = tl.generate_chrome_trace_format(show_memory=False,
        #                                          show_dataflow=True)
        #with open("timeline.json", "w") as f:
        #    f.write(ctf)

        loss_value = o[1]
        
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 5 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, duration))

        if write_summary:
            summary_str = o[2]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 100 == 0:
            checkpoint_path = os.path.join(log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

        # Run validation periodically
        if step > 1 and step % 100 == 0:
            _, top1_error_value = sess.run([val_op, top1_error], { is_training: False })
            print('Validation top1 error %.2f' % top1_error_value)

        if step > 1 and step % 100 == 0:
            g = sess.graph
            i = get_tensors_for_cifar(g)
            o = sess.run(i, { is_training: True })
            with open('tensors.pickle', mode='wb') as f:
                data = {i[idx].name:out for (idx, out) in enumerate(o)}
                pickle.dump(data, f)


