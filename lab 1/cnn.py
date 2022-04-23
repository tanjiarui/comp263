import numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# initial exploration
train_terry, test_terry = dict(), dict()
(train_terry['image'], train_terry['label']), (test_terry['image'], test_terry['label']) = fashion_mnist.load_data()
print(len(train_terry['image']), len(test_terry['image']))
print(train_terry['image'].shape[1:3])
print(max(np.amax(train_terry['image']), np.amax(test_terry['image'])))
print('number of category')
print(np.max(train_terry['label']) + 1)
labels = {0: 'T-shirt', 1: 'trouser', 2: 'pullover', 3: 'dress', 4: 'coat', 5: 'sandal', 6: 'shirt', 7: 'sneaker', 8: 'bag', 9: 'ankle boot'}

def plot_digit(image, label):
	ax = plt.subplot(4, 3, index)
	ax.axis('off')
	ax.set_title(labels[label])
	ax.imshow(image)

# plot images
figure = plt.figure(figsize=(8, 8))
index = 1
for image, label in zip(train_terry['image'][:12], train_terry['label'][:12]):
	plot_digit(image, label)
	index += 1
figure.show()

x_train_terry, x_test_terry, y_train_terry, y_test_terry = train_test_split(train_terry['image'], train_terry['label'], test_size=.2, random_state=48)
# modeling
# sequential
# model = tf.keras.Sequential()
# model.add(tf.keras.Input(shape=(28, 28, 1)))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(100))
# model.add(tf.keras.layers.Dense(len(labels), activation='softmax'))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# functional
def build_model():
	input_layer = tf.keras.Input(shape=(x_train_terry.shape[1], x_train_terry.shape[2], 1))
	layer = tf.keras.layers.BatchNormalization()(input_layer)
	layer = tf.keras.layers.Conv2D(32, 3, activation='relu')(layer)
	layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(layer)
	layer = tf.keras.layers.Conv2D(32, 3, activation='relu')(layer)
	layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(layer)
	layer = tf.keras.layers.Flatten()(layer)
	layer = tf.keras.layers.Dense(100)(layer)
	output_layer = tf.keras.layers.Dense(len(labels), activation='softmax')(layer)

	cnn = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return cnn

model = build_model()
model.summary()
train_history = model.fit(x=x_train_terry, y=y_train_terry, batch_size=256, validation_data=(x_test_terry, y_test_terry), epochs=10)
model.save('cnn')
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

model = tf.keras.models.load_model('cnn')
_, test_acc = model.evaluate(test_terry['image'], test_terry['label'])
print('test accuracy:', round(test_acc, 2))  # test accuracy: 0.89
prediction = model.predict(test_terry['image'])

def plot_distribution(image, target, predict):
	ax = plt.subplot(4, 2, index)
	ax.axis('off')
	ax.set_title(labels[target])
	ax.imshow(image)
	ax = plt.subplot(4, 2, index + 1)
	ax.set_xticks(range(10))
	bar = ax.bar(range(10), predict)
	ax.set_ylim([0, 1])
	predicted_label = np.argmax(predict)
	bar[predicted_label].set_color('blue')
	bar[target].set_color('green')

figure = plt.figure(figsize=(8, 8))
index = 1
for image, label, predict in zip(test_terry['image'][48:52], test_terry['label'][48:52], prediction[48:52]):
	plot_distribution(image, label, list(predict))
	index += 2
figure.show()

prediction = [np.argmax(one_hot) for one_hot in prediction]
print(classification_report(test_terry['label'], prediction))
'''
				precision  recall  f1-score   support
		
		0        0.89       0.78      0.83      1000
		1        0.99       0.98      0.98      1000
		2        0.89       0.79      0.83      1000
		3        0.93       0.87      0.90      1000
		4        0.80       0.90      0.84      1000
		5        0.96       0.98      0.97      1000
		6        0.65       0.76      0.70      1000
		7        0.95       0.95      0.95      1000
		8        0.97       0.97      0.97      1000
		9        0.96       0.96      0.96      1000

accuracy                              0.89     10000
macro avg        0.90      0.89       0.89     10000
weighted avg     0.90      0.89       0.89     10000
'''