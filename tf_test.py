import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

print("可用设备列表：")
for device in tf.config.list_physical_devices():
    print(device)

# 简单线性模型 y = 2x - 1
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]

model.fit(xs, ys, epochs=50, verbose=0)
print("训练后权重：", model.predict([10.0]))  # 应接近 19.0