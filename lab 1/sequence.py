import numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import classification_report

train_terry, test_terry = dict(), dict()
(train_terry['image'], train_terry['label']), (test_terry['image'], test_terry['label']) = fashion_mnist.load_data()
labels = {0: 'T-shirt', 1: 'trouser', 2: 'pullover', 3: 'dress', 4: 'coat', 5: 'sandal', 6: 'shirt', 7: 'sneaker', 8: 'bag', 9: 'ankle boot'}

def build_model():
	input_layer = tf.keras.Input(shape=(28, 28, 1))
	batch_norm = tf.keras.layers.BatchNormalization()(input_layer)
	reshape = tf.keras.layers.Reshape((28, 28))(batch_norm)
	lstm = tf.keras.layers.LSTM(128)(reshape)
	output_layer = tf.keras.layers.Dense(len(labels), activation='softmax')(lstm)

	sequence = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	sequence.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return sequence

model = build_model()
model.summary()
train_history = model.fit(train_terry['image'], train_terry['label'], batch_size=256, validation_split=.2, epochs=10)
model.save('lstm')
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

model = tf.keras.models.load_model('lstm')
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
			precision    recall  f1-score   support

		0       0.78      0.87      0.82      1000
		1       0.99      0.97      0.98      1000
		2       0.81      0.82      0.82      1000
		3       0.86      0.90      0.88      1000
		4       0.81      0.81      0.81      1000
		5       0.97      0.96      0.97      1000
		6       0.74      0.62      0.67      1000
		7       0.94      0.97      0.96      1000
		8       0.98      0.97      0.98      1000
		9       0.98      0.96      0.97      1000

accuracy                            0.89      10000
macro avg       0.89      0.89      0.88      10000
weighted avg    0.89      0.89      0.88      10000
'''