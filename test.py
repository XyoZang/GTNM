import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
a = tf.random.normal([320, 512])
b = tf.random.normal([512, 512])
c = tf.matmul(a, b)
print(sess.run(c).shape)