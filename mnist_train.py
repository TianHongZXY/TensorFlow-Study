import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
#import mnist_inference
import matplotlib.pyplot as plt
import LeNet
# config

batch_size = 100
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularization_rate = 0.0001
training_steps = 10000
moving_average_decay = 0.99
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model.ckpt"

def train(mnist):
	x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
	y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
	regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
	y_output_avg = mnist_inference.inference(x, regularizer)
	global_step = tf.Variable(0, trainable=False)
	variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())



	# cost_function = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_output_avg,labels=tf.argmax(y_,1))
	# cost_function_mean = tf.reduce_mean(cost_function)
	# loss = cost_function + tf.add_n(tf.get_collection('losses'))

	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_output_avg, labels=tf.argmax(y_, 1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

	learning_rate = tf.train.exponential_decay(learning_rate_base,global_step,
	                                           mnist.train.num_examples/batch_size,
	                                           learning_rate_decay,staircase=True)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
	train_op = tf.group(train_step, variable_averages_op)

	correct_pred = tf.equal(tf.argmax(y_output_avg,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		#tf.reset_default_graph()
		for i in range(training_steps):
			xs, ys = mnist.train.next_batch(batch_size)
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs,y_:ys})
			if i % 1000 == 0:
				#validate_acc = sess.run(accuracy, feed_dict=validate_feed)
				print("After %d training steps, loss on training batch is %g" % (i, loss_value))
		# validate_feed = {x: mnist.validation.images,
		#                  y_: mnist.validation.labels}
		saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)
		test_feed = {x: mnist.test.images,
		             y_: mnist.test.labels}
		test_acc = sess.run(accuracy,feed_dict=test_feed)
		print("After %d training steps, test accuracy is %g" % (training_steps, test_acc))

def main(argv=None):
	mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
	train(mnist)
if __name__ == '__main__':
    tf.app.run()