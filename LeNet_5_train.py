import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib.pyplot as plt
import LeNet_5
# config

batch_size = 100
learning_rate_base = 0.0067
learning_rate_decay = 0.99
regularization_rate = 0.0001
training_steps = 5000
moving_average_decay = 0.99
MODEL_SAVE_PATH="LeNet5_model/"
MODEL_NAME="LeNet5_model.ckpt"

def train(mnist):
	x = tf.placeholder(tf.float32, [batch_size, LeNet_5.n_H, LeNet_5.n_W, LeNet_5.n_C], name='x-input')
	y_ = tf.placeholder(tf.float32, [None, LeNet_5.num_labels], name='y-input')
	regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
	global_step = tf.Variable(0, trainable=False)
	variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())

	y_output = LeNet_5.forward_prop(x, regularizer)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_output, labels=tf.argmax(y_, 1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('l2_cost'))

	learning_rate = tf.train.exponential_decay(learning_rate_base,global_step,
	                                           mnist.train.num_examples/batch_size,
	                                           learning_rate_decay,staircase=True)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
	train_op = tf.group(train_step, variable_averages_op)
	correct_pred = tf.equal(tf.argmax(y_output,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		for i in range(training_steps):
			xs, ys = mnist.train.next_batch(batch_size)
			xs = np.reshape(xs, [batch_size,LeNet_5.n_H,LeNet_5.n_W,LeNet_5.n_C])
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs,y_:ys})
			if i % 1000 == 0:
				print("After %d training steps, loss on training batch is %g" % (i, loss_value))

		saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)
		# test_feed = {x:np.reshape(mnist.test.images[:500], [500,LeNet_5.n_H,LeNet_5.n_W,LeNet_5.n_C]),
		#              y_: mnist.test.labels[:500]}
		# test_acc = sess.run(accuracy,feed_dict=test_feed)
		# print("After %d training steps, test accuracy is %g" % (training_steps, test_acc))

def main(argv=None):
	mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
	train(mnist)
if __name__ == '__main__':
    tf.app.run()