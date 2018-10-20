import tensorflow as tf

# config
input_node = 784
output_node = 10
batch_size = 100
n_H = n_W = 28
n_C = 1
num_labels = 10

# conv layer1
conv1_C = 32
conv1_size = 5
# conv layer2
conv2_C = 64
conv2_size = 5
# fc layer
fc2_nodes = 512

x = tf.placeholder(tf.float32, [batch_size, n_H, n_W, n_C], name='x')

def forward_prop(input_tensor, regularizer, train=False):
	# 第一层卷积层
	with tf.variable_scope('layer1-conv1'):
		# 定义过滤器和偏置项
		conv1_filter = tf.get_variable("filter", [conv1_size,conv1_size,n_C,conv1_C],
		                               initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable("bias",[conv1_C],initializer=tf.constant_initializer(0.0))
		# 卷积
		conv1_layer = tf.nn.conv2d(input_tensor,conv1_filter,strides=[1,1,1,1],padding='SAME')
		# 用relu激活，去线性化
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1_layer, conv1_biases))

	# 第二层池化层
	with tf.variable_scope('layer2-pool1'):
		pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	# 第三层卷积层
	with tf.variable_scope('layer3-conv2'):
		conv2_filter = tf.get_variable("filter", [conv2_size, conv2_size, conv1_C, conv2_C],
		                               initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable("bias", [conv2_C], initializer=tf.constant_initializer(0.0))
		conv2_layer = tf.nn.conv2d(pool1, conv2_filter, strides=[1, 1, 1, 1], padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2_layer, conv2_biases))

	# 第四层池化层
	with tf.variable_scope('layer4-pool2'):
		pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# 将第四层输出的池化层[batch_size, 7, 7, 64]reshape成[batch_size,7*7*64]输入给fc1
	pool2_shape = pool2.get_shape().as_list()
	fc1_nodes = pool2_shape[1]*pool2_shape[2]*pool2_shape[3]
	reshaped_pool2 = tf.reshape(pool2, [pool2_shape[0], fc1_nodes])

	# 第五层全连接层
	with tf.variable_scope('layer5-fc1'):
		fc1_weights = tf.get_variable("weights",[fc1_nodes,fc2_nodes],
		                              initializer=tf.truncated_normal_initializer(stddev=0.01))
		if regularizer != None:
			tf.add_to_collection('l2_cost',regularizer(fc1_weights))
		fc1_biases = tf.get_variable('bias',[fc2_nodes],initializer=tf.constant_initializer(0.0))

		fc_layer1 = tf.nn.relu(tf.matmul(reshaped_pool2,fc1_weights) + fc1_biases)
		if train:
			fc_layer1 = tf.nn.dropout(fc_layer1, 0.5)
	# 第六层全连接层（输出层）
	with tf.variable_scope('layer6-fc2'):
		fc2_weights = tf.get_variable("weights", [fc2_nodes, num_labels],
		                              initializer=tf.truncated_normal_initializer(stddev=0.01))
		if regularizer != None:
			tf.add_to_collection('l2_cost', regularizer(fc2_weights))
		fc2_biases = tf.get_variable('bias', [num_labels], initializer=tf.constant_initializer(0.0))

		logit = tf.nn.relu(tf.matmul(fc_layer1, fc2_weights) + fc2_biases)

	return logit
