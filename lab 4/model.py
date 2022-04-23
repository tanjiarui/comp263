import tensorflow as tf, matplotlib.pyplot as plt


class GAN(tf.keras.Model):
	def __init__(self, discriminator, generator):
		super(GAN, self).__init__()
		self.g_optimizer = None
		self.d_optimizer = None
		self.loss_fn = None
		self.g_loss_metric = None
		self.d_loss_metric = None
		self.discriminator = discriminator
		self.generator = generator

	def compile(self, d_optimizer, g_optimizer, loss_fn):
		super(GAN, self).compile()
		self.d_optimizer = d_optimizer
		self.g_optimizer = g_optimizer
		self.loss_fn = loss_fn
		self.d_loss_metric = tf.keras.metrics.Mean(name='d_loss')
		self.g_loss_metric = tf.keras.metrics.Mean(name='g_loss')

	@property
	def metrics(self):
		return [self.d_loss_metric, self.g_loss_metric]

	def train_step(self, real_images):
		# Sample random points in the latent space
		batch_size = tf.shape(real_images)[0]
		random_latent_vectors = tf.random.normal(shape=(batch_size, 100))
		# Decode them to fake images
		generated_images = self.generator(random_latent_vectors)
		# Combine them with real images
		combined_images = tf.concat([generated_images, real_images], axis=0)
		# Assemble labels discriminating real from fake images
		labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
		# Add random noise to the labels - important trick
		labels += 0.05 * tf.random.uniform(tf.shape(labels))
		# Train the discriminator
		with tf.GradientTape() as tape:
			predictions = self.discriminator(combined_images)
			d_loss = self.loss_fn(labels, predictions)
		grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
		self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
		# Sample random points in the latent space
		random_latent_vectors = tf.random.normal(shape=(batch_size, 100))
		# Assemble labels that say 'all real images'
		misleading_labels = tf.zeros((batch_size, 1))
		# Train the generator (note that we should *not* update the weights of the discriminator)
		with tf.GradientTape() as tape:
			predictions = self.discriminator(self.generator(random_latent_vectors))
			g_loss = self.loss_fn(misleading_labels, predictions)
		grads = tape.gradient(g_loss, self.generator.trainable_weights)
		self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
		# Update metrics
		self.d_loss_metric.update_state(d_loss)
		self.g_loss_metric.update_state(g_loss)
		return {'d_loss': self.d_loss_metric.result(), 'g_loss': self.g_loss_metric.result()}


class GANMonitor(tf.keras.callbacks.Callback):
	def __init__(self, num_img=2):
		self.num_img = num_img

	def on_epoch_end(self, epoch, _):
		random_latent_vectors = tf.random.normal(shape=(self.num_img, 100))
		generated_images = self.model.generator(random_latent_vectors)
		generated_images *= 255
		generated_images.numpy()
		figure = plt.figure(figsize=(8, 8))
		plt.suptitle('epoch ' + str(epoch))
		for i in range(self.num_img):
			image = tf.keras.preprocessing.image.array_to_img(tf.reshape(generated_images[i], [generated_images.shape[1], generated_images.shape[2], 1]))
			# plot images
			ax = plt.subplot(4, 4, i + 1)
			ax.axis('off')
			ax.imshow(image)
		figure.show()