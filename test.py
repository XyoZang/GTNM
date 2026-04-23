import tensorflow as tf

print("TF version:", tf.__version__)
print("GPU available:", tf.test.is_gpu_available(cuda_only=True))
print("GPU device name:", tf.test.gpu_device_name())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
a = tf.random.normal([320, 512])
b = tf.random.normal([512, 512])
c = tf.matmul(a, b)
print(sess.run(c).shape)