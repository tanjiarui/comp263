import numpy as np, tensorflow as tf, tensorflow_probability as tfp, matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from keras.layers import Layer

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float') / 255
x_test = x_test.astype('float') / 255
print(x_train.shape[1:3])
print('number of category')
print(np.max(y_train) + 1)


class SampleLayer(Layer):
	def call(self, inputs, *args, **kwargs):
		z_mean, z_log_var = inputs
		batch = tf.shape(z_mean)[0]
		dim = tf.shape(z_mean)[1]
		epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
		return z_mean + tf.exp(.5 * z_log_var) * epsilon


def build_model():
	# encoder
	input_layer = tf.keras.Input(shape=(x_train.shape[1], x_train.shape[2], 1))
	# batch = tf.keras.layers.BatchNormalization(name='batch_norm')(input_layer)
	layer = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(input_layer)
	layer = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(layer)
	layer = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(layer)
	layer = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(layer)
	shape = tf.keras.backend.int_shape(layer)
	layer = tf.keras.layers.Dense(32, activation='relu')(tf.keras.layers.Flatten()(layer))
	z_mean = tf.keras.layers.Dense(2, name='z_mean')(layer)
	z_log_var = tf.keras.layers.Dense(2, name='z_log_var')(layer)
	output_layer = SampleLayer()([z_mean, z_log_var])
	e = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	# decoder
	layer = tf.keras.layers.Dense(shape[1] * shape[2] * shape[3], activation='relu')(e.output)
	layer = tf.keras.layers.Reshape([shape[1], shape[2], shape[3]])(layer)
	layer = tf.keras.layers.Conv2DTranspose(32, 3, activation='relu', padding='same', strides=2)(layer)
	output_layer = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(layer)
	d = tf.keras.Model(inputs=e.output, outputs=output_layer)
	# assemble the encoder and decoder
	v = tf.keras.Model(inputs=e.input, outputs=d(e.output))
	# the input image needs to be normalized before calculate the loss!
	# reconstruct_loss = tf.keras.losses.binary_crossentropy(tf.keras.layers.Flatten()(e.get_layer('batch_norm').output), tf.keras.layers.Flatten()(output_layer))
	reconstruct_loss = tf.keras.losses.binary_crossentropy(tf.keras.layers.Flatten()(e.input), tf.keras.layers.Flatten()(output_layer))
	reconstruct_loss *= x_train.shape[1] * x_train.shape[2]
	kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
	kl_loss = tf.math.reduce_sum(kl_loss, axis=-1)
	kl_loss *= -0.5
	vae_loss = tf.math.reduce_mean(reconstruct_loss + kl_loss)
	v.add_loss(vae_loss)
	v.compile(optimizer='adam')
	return e, d, v


# encoder, decoder, vae = build_model()
# vae.summary()
# vae.fit(x_train, epochs=10, batch_size=256, validation_data=(x_test, None))
# encoder.save('encoder')
# decoder.save('decoder')

encoder, decoder = tf.keras.models.load_model('encoder'), tf.keras.models.load_model('decoder')
# generate 10x10 samples
n = 10
figure_size = 28
norm = tfp.distributions.Normal(0, 1)
grid_x = norm.quantile(np.linspace(.05, .95, n))
grid_y = norm.quantile(np.linspace(.05, .95, n))
figure = np.zeros((figure_size * n, figure_size * n))
for i, yi in enumerate(grid_x):
	for j, xi in enumerate(grid_y):
		z_sample = np.array([[xi, yi]])
		z_sample = np.tile(z_sample, 256).reshape(256, 2)
		decode = decoder.predict(z_sample, batch_size=256)
		image = decode[0].reshape(figure_size, figure_size)
		figure[i * figure_size: (i + 1) * figure_size, j * figure_size: (j + 1) * figure_size] = image
plt.figure(figsize=(20, 20))
plt.imshow(figure)
plt.show()
# latent variable visualization
z = encoder.predict(x_test)
plt.scatter(x=z[:, 0], y=z[:, 1], c=y_test)
plt.colorbar()
plt.show()