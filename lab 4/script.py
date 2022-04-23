import time, numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler
from model import *

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
shape = [-1, x_train.shape[1], x_train.shape[2]]
x_train = StandardScaler().fit_transform(x_train.astype('float').reshape(-1, 1)).reshape(shape)
x_test = StandardScaler().fit_transform(x_test.astype('float').reshape(-1, 1)).reshape(shape)
print(x_train.shape[1:3])
print('number of category')
print(np.max(y_train) + 1)
label = np.where(y_train == 1)[0]
x_train = x_train[label]
label = np.where(y_test == 1)[0]
x_test = x_test[label]
dataset = np.concatenate([x_train, x_test], dtype='float')
print(dataset.shape)
# plot images
figure = plt.figure(figsize=(8, 8))
index = 1
for image in dataset[:12]:
	ax = plt.subplot(4, 3, index)
	ax.axis('off')
	ax.imshow(image)
	index += 1
figure.show()


def generator():
	noise = tf.keras.Input(shape=100)
	layer = tf.keras.layers.Dense(units=7 * 7 * 256, use_bias=False)(noise)
	layer = tf.keras.layers.BatchNormalization()(layer)
	layer = tf.keras.layers.LeakyReLU()(layer)
	layer = tf.keras.layers.Reshape(target_shape=(7, 7, 256))(layer)
	layer = tf.keras.layers.Conv2DTranspose(128, 5, padding='same', use_bias=False)(layer)
	layer = tf.keras.layers.BatchNormalization()(layer)
	layer = tf.keras.layers.LeakyReLU()(layer)
	layer = tf.keras.layers.Conv2DTranspose(64, 5, padding='same', strides=2, use_bias=False)(layer)
	layer = tf.keras.layers.BatchNormalization()(layer)
	layer = tf.keras.layers.LeakyReLU()(layer)
	layer = tf.keras.layers.Conv2DTranspose(1, 5, activation='tanh', padding='same', strides=2, use_bias=False)(layer)
	output_layer = tf.keras.layers.Reshape(target_shape=(dataset.shape[1], dataset.shape[2]))(layer)
	model = tf.keras.Model(inputs=noise, outputs=output_layer, name='generator')
	model.summary()
	return model


def discriminator():
	input_layer = tf.keras.Input(shape=(dataset.shape[1], dataset.shape[2], 1))
	layer = tf.keras.layers.Conv2D(64, 5, padding='same', strides=2)(input_layer)
	layer = tf.keras.layers.LeakyReLU()(layer)
	layer = tf.keras.layers.Dropout(.3)(layer)
	layer = tf.keras.layers.Conv2D(128, 5, padding='same', strides=2)(layer)
	layer = tf.keras.layers.LeakyReLU()(layer)
	layer = tf.keras.layers.Dropout(.3)(layer)
	layer = tf.keras.layers.Conv2DTranspose(64, 5, padding='same', strides=2, use_bias=False)(layer)
	layer = tf.keras.layers.BatchNormalization()(layer)
	layer = tf.keras.layers.LeakyReLU()(layer)
	layer = tf.keras.layers.Flatten()(layer)
	output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(layer)
	model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='discriminator')
	model.summary()
	return model


epochs = 10
gan = GAN(discriminator=discriminator(), generator=generator())
gan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss_fn=tf.keras.losses.BinaryCrossentropy())
start = time.time()
gan.fit(dataset, epochs=epochs, callbacks=[GANMonitor(num_img=16)])
end = time.time()
print('training duration: %.2f seconds' % (end - start))