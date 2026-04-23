import tensorflow as tf
import time

print("1. Import done")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

print("2. Creating session...")
sess = tf.Session(config=config)
print("3. Session created!")

a = tf.random.normal([10, 10])
b = tf.random.normal([10, 10])
c = tf.matmul(a, b)
print("4. Running matmul...")
result = sess.run(c)
print("5. Result shape:", result.shape)
sess.close()
print("6. Done!")