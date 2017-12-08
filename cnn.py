# More Advanced CNN Model: CIFAR-10
# ---------------------------------------
#
# In this example, we will download the CIFAR-10 images
# and build a CNN model with dropout and regularization
#
# CIFAR is composed ot 50k train and 10k test
# images that are 32x32.

import os
import sys
import csv
import tensorflow as tf
from tensorflow.python.framework import ops
from progressbar import ProgressBar
import random
from visualize_conv import put_kernels_on_grid

ops.reset_default_graph()
random.seed()

DATASET_DIR = "dataset"
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
LABELS_CSV = "labels.csv"


def main(options, args):
    # Change Directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # read label
    label_list = list(csv.reader(open(os.path.join(DATASET_DIR, LABELS_CSV))))

    # Set model parameters
    # CNNの各層の特徴量
    seed = options.seed
    features1 = options.features1
    features2 = options.features2
    data_dir = options.data_directory
    generations = options.generations
    output_every = options.output_every
    eval_every = options.eval_every
    batch_size = 128
    image_size = 100
    crop_size = 50
    num_channels = 3
    num_targets = len(label_list)

    # Exponential Learning Rate Decay Params
    learning_rate = 0.1
    lr_decay = 0.1
    num_gens_to_wait = 250.

    # Check data directory
    if not os.path.exists(data_dir):
        sys.exit('%s is not directory' % data_dir)

    # Set random seed
    tf.set_random_seed(seed)

    # Start a graph session
    sess = tf.Session()

    # Tensorboard用のログ出力
    summary_dir = 'logs/{}'.format(seed)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    writer = tf.summary.FileWriter(summary_dir, graph_def=sess.graph_def)

    # Define reader
    def read_image_files(csv, shuffle=True, distort_images=True):
        queue = tf.train.string_input_producer(csv, shuffle=shuffle)
        reader = tf.TextLineReader()
        key, record_string = reader.read(queue)
        filename, image_label = tf.decode_csv(record_string, [["file_path"], [1]])

        jpeg = tf.read_file(filename)
        image_extracted = tf.image.decode_jpeg(jpeg, channels=3)
        image_extracted.set_shape([image_size, image_size, 3])
        final_image = tf.cast(image_extracted, tf.float32)
        tf.summary.image('training images(Pre)', tf.reshape(final_image, [-1, image_size, image_size, num_channels]))
        # Randomly Crop image
        final_image = tf.image.resize_images(final_image, [crop_size, crop_size])

        if distort_images:
            # Randomly flip the image horizontally, change the brightness and contrast
            tf.image.random_flip_left_right(final_image)

        # Normalize whitening
        tf.image.per_image_standardization(final_image)

        # Randomly resize with crop or pad

        tf.summary.image('training images(After)', tf.reshape(final_image, [-1, crop_size, crop_size, num_channels]))
        return (final_image, image_label)

    # Create a image pipeline from reader
    def input_pipeline(batch_size, train_logical=True):
        if train_logical:
            csvname = [os.path.join(data_dir, TRAIN_CSV)]
        else:
            csvname = [os.path.join(data_dir, TEST_CSV)]
        image, label = read_image_files(csvname)

        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 5000
        capacity = min_after_dequeue + 3 * batch_size
        example_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                            batch_size=batch_size,
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)

        return (example_batch, label_batch)

    # Define the model architecture, this will return logits from images
    def da_face_cnn_model(input_images, batch_size, train_logical=True):
        def truncated_normal_var(name, shape, dtype):
            return (tf.get_variable(name=name, shape=shape, dtype=dtype,
                                    initializer=tf.truncated_normal_initializer(stddev=0.05)))

        def zero_var(name, shape, dtype):
            return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

        # First Convolutional Layer
        with tf.variable_scope('conv1') as scope:
            # Conv_kernel is 5x5 for all 3 colors and we will create 64 features
            conv1_kernel = truncated_normal_var(name='conv_kernel1', shape=[5, 5, 3, features1], dtype=tf.float32)
            tf.summary.image('conv1_kernel', put_kernels_on_grid(conv1_kernel))
            # We convolve across the image with a stride size of 1
            conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1, 1, 1, 1], padding='SAME')
            # Initialize and add the bias term
            conv1_bias = zero_var(name='conv_bias1', shape=[features1], dtype=tf.float32)
            conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
            # ReLU element wise
            relu_conv1 = tf.nn.relu(conv1_add_bias)
            # tf.summary.histogram('conv1', relu_conv1)

        # Max Pooling
        pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer1')

        # Local Response Normalization (parameters from paper)
        # paper: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
        norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')

        # Second Convolutional Layer
        with tf.variable_scope('conv2') as scope:
            # Conv kernel is 5x5, across all prior features and we create more features
            conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5, 5, features1, features2],
                                                dtype=tf.float32)
            # Convolve filter across prior output with stride size of 1
            conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
            # Initialize and add the bias
            conv2_bias = zero_var(name='conv_bias2', shape=[features2], dtype=tf.float32)
            conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
            tf.summary.histogram('conv2', conv2_add_bias)
            # ReLU element wise
            relu_conv2 = tf.nn.relu(conv2_add_bias)

        # Max Pooling
        pool2 = tf.nn.max_pool(relu_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer2')

        # Local Response Normalization (parameters from paper)
        norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')

        # Reshape output into a single matrix for multiplication for the fully connected layers
        reshaped_output = tf.reshape(norm2, [batch_size, -1])
        reshaped_dim = reshaped_output.get_shape()[1].value

        # First Fully Connected Layer
        with tf.variable_scope('full1') as scope:
            # Fully connected layer will have 384 outputs.
            full_weight1 = truncated_normal_var(name='full_mult1', shape=[reshaped_dim, 384], dtype=tf.float32)
            full_bias1 = zero_var(name='full_bias1', shape=[384], dtype=tf.float32)
            full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weight1), full_bias1))

        # Second Fully Connected Layer
        with tf.variable_scope('full2') as scope:
            # Second fully connected layer has 192 outputs.
            full_weight2 = truncated_normal_var(name='full_mult2', shape=[384, 192], dtype=tf.float32)
            full_bias2 = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)
            full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))

        # Final Fully Connected Layer -> categories for output (num_targets)
        with tf.variable_scope('full3') as scope:
            # Final fully connected layer has num_targets outputs.
            full_weight3 = truncated_normal_var(name='full_mult3', shape=[192, num_targets], dtype=tf.float32)
            full_bias3 = zero_var(name='full_bias3', shape=[num_targets], dtype=tf.float32)
            final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)
        return (final_output)

    # Loss function
    def da_face_loss(logits, targets):
        # Get rid of extra dimensions and cast targets into integers
        targets = tf.squeeze(tf.cast(targets, tf.int32))
        # Calculate cross entropy from logits and targets
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        # Take the average loss across batch size
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.summary.scalar('loss', cross_entropy_mean)
        return (cross_entropy_mean)

    # Train step
    def train_step(loss_value, generation_num):
        # Our learning rate is an exponential decay after we wait a fair number of generations
        model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num,
                                                         num_gens_to_wait, lr_decay, staircase=True)
        # Create optimizer
        my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
        # Initialize train step
        train_step = my_optimizer.minimize(loss_value)
        return (train_step)

    # Accuracy function
    def accuracy_of_batch(logits, targets):
        # Make sure targets are integers and drop extra dimensions
        targets = tf.squeeze(tf.cast(targets, tf.int32))
        # Get predicted values by finding which logit is the greatest
        batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
        # Check if they are equal across the batch
        predicted_correctly = tf.equal(batch_predictions, targets)
        # Average the 1's and 0's (True's and False's) across the batch size
        accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return (accuracy)

    # Get data
    print('Getting/Transforming Data.')
    # Initialize the data pipeline
    images, targets = input_pipeline(batch_size, train_logical=True)
    # Get batch test images and targets from pipline
    test_images, test_targets = input_pipeline(batch_size, train_logical=False)

    # Declare Model
    print('Creating the DA Face Model.')
    with tf.variable_scope('model_definition') as scope:
        # Declare the training network model
        model_output = da_face_cnn_model(images, batch_size)
        # This is very important!!!  We must set the scope to REUSE the variables,
        #  otherwise, when we set the test network model, it will create new random
        #  variables.  Otherwise we get random evaluations on the test batches.
        scope.reuse_variables()
        test_output = da_face_cnn_model(test_images, batch_size)
        test_probability = tf.nn.softmax(test_output)

    # Declare loss function
    print('Declare Loss Function.')
    loss = da_face_loss(model_output, targets)

    # Create accuracy function
    accuracy = accuracy_of_batch(test_output, test_targets)

    # Create training operations
    print('Creating the Training Operation.')
    generation_num = tf.Variable(0, trainable=False)
    train_op = train_step(loss, generation_num)

    # Initialize Variables
    print('Initializing the Variables.')
    init = tf.global_variables_initializer()
    sess.run(init)

    # Initialize queue (This queue will feed into the model, so no placeholders necessary)
    tf.train.start_queue_runners(sess=sess)

    # Train CIFAR Model
    print('Starting Training. Seed:{}'.format(seed))
    print('Generations:{}, Output every:{}, Eval every:{}'.format(generations, output_every, eval_every))
    print('Convolutional layers features. f1:{}, f2:{}'.format(features1, features2))
    # Tensorboard
    summary = tf.summary.merge_all()
    train_loss = []
    test_accuracy = []

    def printLossValue(loss_value):
        train_loss.append(loss_value)
        output = ' Generation {}: Loss = {:.5f}'.format((i + 1), loss_value)
        print(output)

    def printTestAccuracy(temp_accuracy):
        test_accuracy.append(temp_accuracy)
        acc_output = ' --- Test Accuracy = {:.2f}%.'.format(100. * temp_accuracy)
        print(acc_output)

    # def printTestOutput(target, test_probability):
    #     labels = {
    #         "other": 0,
    #         "simamura": 1,
    #         "ckobayashi": 2,
    #         "nagadou": 3,
    #         "kato": 4,
    #         "inoue": 5,
    #         "sasaki": 6,
    #         "wada": 7,
    #         "takahashi": 8,
    #         "isegawa": 9,
    #         "ishida": 10,
    #         "sakurai": 11,
    #         "Arnold_Schwarzenegger": 12,
    #         "fujimoto": 13,
    #         "kinjo": 14,
    #         "quy": 15,
    #         "takebayasi": 16,
    #         "yada": 17,
    #         "yanada": 18,
    #         "yasukawa": 19,
    #     }
    #
    #     target_prob = ''
    #     other_probs = []
    #     for label in labels:
    #         label_name = labels[label]
    #         if target == label_name:
    #             target_prob = ' {}:{:.2f}%'.format(label, 100 * test_probability[target])
    #         else:
    #             other_probs.append('{}:{:.2f}%'.format(label, 100 * test_probability[label_name]))
    #     print(target_prob)
    #     print(other_probs)

    p = ProgressBar()
    p(range(generations))
    for i in range(generations):
        _, loss_value, w_summary, t_tar, t_prob = sess.run([train_op, loss, summary, test_targets, test_probability])
        p.update(i + 1)

        if (i + 1) % output_every == 0:
            printLossValue(loss_value)

        if (i + 1) % eval_every == 0:
            [temp_accuracy] = sess.run([accuracy])
            printTestAccuracy(temp_accuracy)
            # printTestOutput(t_tar[0], t_prob[0])

        if loss_value < 0.0001:
            printLossValue(loss_value)
            printTestAccuracy()
            break
        writer.add_summary(w_summary, i)


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-s', '--seed', dest='seed', action='store', type='int', default=random.getrandbits(30))
    parser.add_option('-g', '--generations', dest='generations', action='store', type='int', default=20000)
    parser.add_option('-o', '--output-every', dest='output_every', action='store', type='int', default=100)
    parser.add_option('-e', '--eval-every', dest='eval_every', action='store', type='int', default=100)
    parser.add_option('-d', '--data-directory', dest='data_directory', action='store', type='string',
                      default=DATASET_DIR)
    parser.add_option('--features1', dest='features1', action='store', type='int', default=64)
    parser.add_option('--features2', dest='features2', action='store', type='int', default=64)
    options, args = parser.parse_args(sys.argv[1:])

    main(options, args)
