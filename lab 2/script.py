import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

unsupervised, supervised = dict(), dict()
(unsupervised['image'], _), (supervised['image'], supervised['label']) = fashion_mnist.load_data()

print(len(unsupervised['image']), len(supervised['image']))
print(unsupervised['image'].shape[1:3])
print(max(np.amax(unsupervised['image']), np.amax(supervised['image'])))
print('number of category')
print(np.max(supervised['label']) + 1)

# split supervised dataset
x_train, x_test, y_train, y_test = train_test_split(supervised['image'][:3000] / 255, supervised['label'][:3000], test_size=.4, random_state=48)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=.5, random_state=48)
labels = {0: 'T-shirt', 1: 'trouser', 2: 'pullover', 3: 'dress', 4: 'coat', 5: 'sandal', 6: 'shirt', 7: 'sneaker', 8: 'bag', 9: 'ankle boot'}

# baseline model
def build_baseline():
	input_layer = tf.keras.Input(shape=(x_train.shape[1], x_train.shape[2], 1))
	layer = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', strides=2)(input_layer)
	layer = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same', strides=2)(layer)
	layer = tf.keras.layers.Flatten()(layer)
	layer = tf.keras.layers.Dense(100)(layer)
	output_layer = tf.keras.layers.Dense(len(labels), activation='softmax')(layer)

	cnn = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return cnn

model = build_baseline()
model.summary()
train_history = model.fit(x=x_train, y=y_train, batch_size=256, validation_data=(x_test, y_test), epochs=10)
model.save('baseline')
# plot metric
epoch = train_history.epoch
train_accuracy = train_history.history['accuracy']
val_accuracy = train_history.history['val_accuracy']
plt.title('train vs validation')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.plot(epoch, train_accuracy, color='b', label='train')
plt.plot(epoch, val_accuracy, color='r', label='validation')
plt.legend()
plt.show()
# validation
model = tf.keras.models.load_model('baseline')
prediction = [np.argmax(one_hot) for one_hot in model.predict(x_val)]
print(classification_report(y_val, prediction))
'''
			precision    recall   f1-score    support

	0           0.79      0.86      0.82        65
	1           0.85      0.86      0.85        58
	2           0.78      0.47      0.58        45
	3           0.82      0.67      0.74        67
	4           0.49      0.89      0.63        64
	5           0.97      0.42      0.59        66
	6           0.57      0.24      0.33        55
	7           0.73      0.82      0.77        71
	8           0.82      0.91      0.86        56
	9           0.66      0.98      0.79        53

accuracy                            0.72       600
macro avg       0.75      0.71      0.70       600
weighted avg    0.75      0.72      0.70       600
'''

# split unsupervised dataset
train_pure, test_pure = train_test_split(unsupervised['image'].astype('float') / 255, test_size=.05, random_state=48)
noise = np.random.normal(0, 1, train_pure.shape)
train_noise = train_pure + noise * .2
noise = np.random.normal(0, 1, test_pure.shape)
test_noise = test_pure + noise * .2

# autoencoder
def build_autoencoder():
	input_layer = tf.keras.Input(shape=(train_noise.shape[1], train_noise.shape[2], 1))
	encoder = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', strides=2)(input_layer)
	encoder = tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same', strides=2)(encoder)
	decoder = tf.keras.layers.Conv2DTranspose(8, 3, activation='relu', padding='same', strides=2)(encoder)
	decoder = tf.keras.layers.Conv2DTranspose(16, 3, activation='relu', padding='same', strides=2)(decoder)
	decoder = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(decoder)

	autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
	autoencoder.compile(optimizer='adam', loss='mean_squared_error')
	return autoencoder

model = build_autoencoder()
model.summary()
train_history = model.fit(x=train_noise, y=train_pure, batch_size=256, validation_data=(test_noise, test_pure), epochs=10)
model.save('autoencoder')

model = tf.keras.models.load_model('autoencoder')
prediction = model.predict(test_noise)
index = 1
for image in prediction[:10]:
	image = np.mean(image, axis=2)
	ax = plt.subplot(5, 2, index)
	ax.axis('off')
	ax.imshow(image)
	index += 1
plt.show()

encoder = model.layers[:3]
# transfer learning
def transfer():
	input_layer = encoder[0].input
	layer = encoder[1](input_layer)
	layer = encoder[2](layer)
	layer = tf.keras.layers.Flatten()(layer)
	layer = tf.keras.layers.Dense(100)(layer)
	output_layer = tf.keras.layers.Dense(len(labels), activation='softmax')(layer)

	transfer_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	transfer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return transfer_model

model = transfer()
model.summary()
train_history = model.fit(x=x_train, y=y_train, batch_size=256, validation_data=(x_test, y_test), epochs=10)
model.save('transfer model')
# plot metric
epoch = train_history.epoch
train_accuracy = train_history.history['accuracy']
val_accuracy = train_history.history['val_accuracy']
plt.title('train vs validation')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.plot(epoch, train_accuracy, color='b', label='train')
plt.plot(epoch, val_accuracy, color='r', label='validation')
plt.legend()
plt.show()
# validation
model = tf.keras.models.load_model('transfer model')
prediction = [np.argmax(one_hot) for one_hot in model.predict(x_val)]
print(classification_report(y_val, prediction))
'''
			precision    recall   f1-score    support

	0           0.71      0.86      0.78        65
	1           0.87      0.91      0.89        58
	2           0.85      0.38      0.52        45
	3           0.85      0.67      0.75        67
	4           0.52      0.84      0.64        64
	5           0.83      0.80      0.82        66
	6           0.44      0.33      0.37        55
	7           0.89      0.80      0.84        71
	8           0.92      0.79      0.85        56
	9           0.76      0.94      0.84        53

accuracy                            0.74       600
macro avg       0.76      0.73      0.73       600
weighted avg    0.76      0.74      0.74       600
'''